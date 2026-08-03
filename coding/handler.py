"""The RunPod serverless entry point for the coding worker: an OpenAI-route passthrough.

── The contract ──────────────────────────────────────────────────────────────────────────────────

A request to `https://api.runpod.ai/v2/<endpoint-id>/openai/<path>` reaches the handler as:

```jsonc
{"input": {
  "openai_route": "/v1/messages/count_tokens",   // everything after `/openai`
  "openai_input": { /* the request body, as a dict */ }
}}
```

RunPod does not proxy that path anywhere — it hands it to the worker, and **the worker decides which
routes exist**. That single fact is why this file is a passthrough and not a dispatch table.

── Why a passthrough, and not an allowlist ───────────────────────────────────────────────────────

RunPod's own `worker-v1-vllm` dispatches on a closed set: Chat Completions, Models, Responses and
Messages. Anything else raises inside the handler, which RunPod turns into a **500** — and measured
on the live endpoint on 2026-08-03, `/v1/messages/count_tokens` returned a response byte-for-byte
identical to a route that does not exist at all:

    /v1/models                    HTTP 200
    /v1/this-route-does-not-exist HTTP 500  {"status":500,"title":"Internal Server Error",…}
    /v1/messages/count_tokens     HTTP 500  {"status":500,"title":"Internal Server Error",…}

That is not a missing feature in vLLM — vLLM's Python frontend serves `count_tokens` — it is a route
the worker never forwarded. The consequence for a client is worse than a 404: the Anthropic SDK
treats 5xx as **retryable**, so Claude Code hammered the route ten times with backoff and stalled,
where a 404 would have failed fast into its own token estimate.

So this handler forwards whatever route it is given to the local vLLM and returns whatever comes
back. New upstream routes work the day the base image ships them, with no change here.

── Errors are returned, never raised ─────────────────────────────────────────────────────────────

A handler that raises takes the worker down with it, and RunPod's scaler reads a dying worker as a
broken endpoint rather than as one bad request. Everything below returns an OpenAI-shaped error
object instead, so a client can tell "you asked for something that does not exist" apart from "the
GPU fell over".
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, Final

from . import vllm_server as coding_server
from .config import Config, ConfigError, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log: Final = logging.getLogger("coding")

# Routes that carry no body and must be sent as GET. Everything else is a POST — which is correct
# for the entire generative surface (`/v1/chat/completions`, `/v1/completions`, `/v1/messages`,
# `/v1/messages/count_tokens`, `/v1/responses`, `/v1/embeddings`, …).
#
# Prefix-matched rather than exact so `/v1/models/<id>` is covered without enumerating model names.
_GET_PREFIXES: Final = ("/v1/models", "/health", "/version", "/ping", "/metrics")

# vLLM answers SSE as `data: {...}\n\n` lines. RunPod expects each yielded value to be one chunk of
# the stream it will re-emit to the client, so the framing is preserved exactly rather than parsed
# and rebuilt — a re-serialised chunk is a chance to differ from what the engine actually said.
_SSE_ENCODING: Final = "utf-8"


def normalise_route(raw: Any) -> str:
    """`v1/models` and `/v1/models` are the same route; anything else is a client error."""
    if not isinstance(raw, str) or raw.strip() == "":
        raise ValueError("the job carried no `openai_route`")
    route = raw.strip()
    if not route.startswith("/"):
        route = "/" + route
    # A route is a path, not a URL. Accepting one would let a caller aim this worker's HTTP client
    # at an arbitrary host from inside the container.
    if "://" in route or route.startswith("//"):
        raise ValueError(f"openai_route must be a path, got {raw!r}")
    return route


def is_get(route: str) -> bool:
    return any(route == prefix or route.startswith(prefix + "/") for prefix in _GET_PREFIXES)


def error_payload(message: str, kind: str = "invalid_request_error", code: str | None = None) -> dict[str, Any]:
    """The OpenAI error envelope, which is also what the Anthropic SDK's error parser understands."""
    return {"error": {"message": message, "type": kind, "code": code}}


def _client(config: Config):  # noqa: ANN202 — httpx is imported lazily; see the note in pyproject.
    import httpx

    return httpx.Client(
        base_url=f"http://{config.vllm_host}:{config.vllm_port}",
        timeout=config.request_timeout_s,
    )


def forward(route: str, body: dict[str, Any], config: Config) -> dict[str, Any]:
    """One non-streaming request to the local vLLM, returned verbatim.

    A non-2xx is returned as a body rather than raised: vLLM's own error envelope is more useful to
    a client than anything this layer could invent, and raising would cost the worker.
    """
    with _client(config) as client:
        response = client.get(route) if is_get(route) else client.post(route, json=body)
        try:
            return response.json()
        except ValueError:
            # vLLM answered with something that is not JSON — a proxy error page, or a truncated
            # body. Surfacing the status and a prefix beats a decoder traceback with no context.
            return error_payload(
                f"vllm answered {response.status_code} with a non-JSON body: {response.text[:400]!r}",
                kind="api_error",
                code=str(response.status_code),
            )


def forward_stream(route: str, body: dict[str, Any], config: Config) -> Iterator[str]:
    """One streaming request, re-emitted chunk by chunk exactly as vLLM framed it.

    `iter_raw` rather than `iter_lines`: the SSE framing — the blank line between events, the
    `data:` prefix — is part of what the client parses, and re-assembling it from decoded lines is
    a chance to differ from what the engine actually said.
    """
    with _client(config) as client, client.stream("POST", route, json=body) as response:
        if response.status_code >= 400:
            response.read()
            payload = error_payload(response.text[:400], kind="api_error", code=str(response.status_code))
            yield f"data: {json.dumps(payload)}\n\n"
            return
        for chunk in response.iter_raw():
            if chunk:
                yield chunk.decode(_SSE_ENCODING, errors="replace")


def parse_job(job: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """The RunPod envelope in, `(route, body)` out. Raises `ValueError` on anything malformed.

    Separated from `handler` so the validation cascade is one function with one failure mode,
    instead of a chain of early returns that has to remember to build an error envelope at each step.
    """
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        raise ValueError("the job carried no `input` object")

    raw_route = job_input.get("openai_route")
    if raw_route is None:
        # Reached by a POST to `/run` or `/runsync` rather than to `/openai/...`. Naming the correct
        # URL here is worth more than a generic rejection: it is the single most likely mistake a
        # new client makes against this endpoint.
        raise ValueError(
            "this endpoint speaks only the OpenAI-compatible surface. Post to "
            "https://api.runpod.ai/v2/<endpoint-id>/openai/v1/... rather than to /run or /runsync, "
            "and the route will arrive here as `openai_route`."
        )

    body = job_input.get("openai_input")
    if body is None:
        # `/v1/models` arrives with no body at all. That is a GET, not a client error.
        body = {}
    if not isinstance(body, dict):
        raise ValueError("`openai_input` must be an object")

    return normalise_route(raw_route), body


def make_handler(config: Config):  # noqa: ANN201 — the RunPod SDK types its handler as a bare callable.
    def handler(job: dict[str, Any]) -> Any:
        try:
            route, body = parse_job(job)
        except ValueError as err:
            return error_payload(str(err))

        try:
            # A GET is never streamed: `stream: true` on `/v1/models` is nonsense, and asking httpx
            # to stream a request with no response body hangs until the read timeout.
            if body.get("stream") is True and not is_get(route):
                return forward_stream(route, body, config)
            return forward(route, body, config)
        # Broad on purpose: see the module docstring. A raising handler takes the worker with it.
        except Exception as err:
            log.exception("coding: request to %s failed", route)
            return error_payload(f"{type(err).__name__}: {err}", kind="api_error")

    return handler


def main() -> None:
    import runpod

    config = load_config()
    log.info(
        "coding: model=%s reasoning_parser=%r tool_call_parser=%r",
        config.model_id,
        config.reasoning_parser,
        config.tool_call_parser,
    )
    # Before the handler is registered, and blocking — see `common.vllm_server`. FlashBoot snapshots
    # a worker once it is idle, so a model that loads lazily is a model that is never in the snapshot.
    coding_server.start(config)

    runpod.serverless.start(
        {
            "handler": make_handler(config),
            # 🔴 **Without this a worker takes ONE job at a time**, and the whole point of vLLM's
            # continuous batching is lost. Measured against RunPod's own worker on 2026-08-03: three
            # simultaneous 150-token requests on a single worker returned in 4.0 s total, against
            # 3.8–4.3 s for the same request sent alone — the batch is effectively free. A coding
            # harness fires the main turn, its background summarisation calls and any subagent at
            # once, so serialising them would show up as latency on every single turn.
            #
            # The ceiling is deliberately not "unbounded": each in-flight request holds KV cache,
            # and past what the cache can hold vLLM preempts and recomputes rather than refusing.
            "concurrency_modifier": lambda _current: config.max_concurrency,
            # Generator handlers need this to also work for non-streaming callers of /runsync.
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    try:
        main()
    except ConfigError as err:
        log.error("coding: %s", err)
        raise SystemExit(2) from err
