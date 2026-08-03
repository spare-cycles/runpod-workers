"""The passthrough's contract, exercised without vLLM, httpx or a GPU.

Everything here is reachable on a CPU runner because the HTTP client is imported lazily inside the
two forwarding functions. The tests that would need a live engine — that vLLM really answers
`/v1/messages/count_tokens` — belong on the endpoint, not here.
"""

from __future__ import annotations

import pytest

from coding import handler
from coding.config import load_config

BASE_ENV = {"WORKER_MODEL_ID": "Qwen/Qwen3.6-27B-FP8"}


def config():
    return load_config(BASE_ENV)


# ── Route normalisation ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("/v1/models", "/v1/models"),
        ("v1/models", "/v1/models"),
        ("  /v1/messages/count_tokens  ", "/v1/messages/count_tokens"),
    ],
)
def test_normalise_route_accepts_both_forms(raw, expected):
    assert handler.normalise_route(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", 7, ["/v1/models"]])
def test_normalise_route_rejects_non_routes(raw):
    with pytest.raises(ValueError):
        handler.normalise_route(raw)


@pytest.mark.parametrize("raw", ["http://evil.example/v1/models", "//evil.example/x", "https://x/y"])
def test_normalise_route_rejects_absolute_urls(raw):
    """A route is a path. Accepting a URL would aim the worker's HTTP client at an arbitrary host."""
    with pytest.raises(ValueError):
        handler.normalise_route(raw)


# ── GET vs POST ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("route", ["/v1/models", "/v1/models/Qwen/Qwen3.6-27B-FP8", "/health", "/metrics"])
def test_bodyless_routes_are_get(route):
    assert handler.is_get(route)


@pytest.mark.parametrize(
    "route",
    [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        # The route this whole worker exists for: RunPod's own worker 500s on it because it is not
        # in that worker's closed dispatch table. Here it is simply a POST like any other.
        "/v1/messages/count_tokens",
        "/v1/responses",
        "/v1/embeddings",
    ],
)
def test_generative_routes_are_post(route):
    assert not handler.is_get(route)


def test_models_prefix_does_not_swallow_a_similarly_named_route():
    """`/v1/modelsomething` is not under `/v1/models`, and must not be turned into a GET."""
    assert not handler.is_get("/v1/modelsomething")


# ── The job envelope ──────────────────────────────────────────────────────────────────────────


def test_missing_input_object_is_an_error_not_a_raise():
    result = handler.make_handler(config())({})
    assert "error" in result


def test_a_run_style_job_is_told_which_url_to_use():
    """The likeliest client mistake — posting to /run instead of /openai/... — names its own fix."""
    result = handler.make_handler(config())({"input": {"prompt": "hi"}})
    assert "/openai/v1/" in result["error"]["message"]


def test_non_object_openai_input_is_rejected():
    job = {"input": {"openai_route": "/v1/messages", "openai_input": "not a dict"}}
    result = handler.make_handler(config())(job)
    assert "error" in result


def test_forwarding_failure_is_returned_not_raised(monkeypatch):
    """A handler that raises takes the worker down with it, and the scaler reads that as broken."""

    def boom(*_args, **_kwargs):
        raise RuntimeError("vllm is not listening")

    monkeypatch.setattr(handler, "forward", boom)
    job = {"input": {"openai_route": "/v1/messages", "openai_input": {"model": "x"}}}
    result = handler.make_handler(config())(job)
    assert result["error"]["type"] == "api_error"
    assert "vllm is not listening" in result["error"]["message"]


def test_streaming_requests_take_the_streaming_path(monkeypatch):
    monkeypatch.setattr(handler, "forward_stream", lambda *_a, **_k: iter(["data: {}\n\n"]))
    monkeypatch.setattr(handler, "forward", lambda *_a, **_k: {"not": "used"})
    job = {"input": {"openai_route": "/v1/messages", "openai_input": {"model": "x", "stream": True}}}
    assert list(handler.make_handler(config())(job)) == ["data: {}\n\n"]


def test_a_get_route_is_never_streamed(monkeypatch):
    """`stream: true` on `/v1/models` is nonsense, and streaming a GET would hang on a body."""
    monkeypatch.setattr(handler, "forward", lambda *_a, **_k: {"object": "list"})
    monkeypatch.setattr(handler, "forward_stream", lambda *_a, **_k: pytest.fail("streamed a GET"))
    job = {"input": {"openai_route": "/v1/models", "openai_input": {"stream": True}}}
    assert handler.make_handler(config())(job) == {"object": "list"}


def test_absent_openai_input_defaults_to_an_empty_body(monkeypatch):
    """`/v1/models` arrives with no body at all; that is a GET, not a client error."""
    seen = {}
    monkeypatch.setattr(handler, "forward", lambda route, body, _c: seen.update(route=route, body=body) or {"ok": 1})
    handler.make_handler(config())({"input": {"openai_route": "/v1/models"}})
    assert seen == {"route": "/v1/models", "body": {}}
