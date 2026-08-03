"""Every knob the coding worker reads, resolved once at start-up.

Everything here is configuration rather than a request field for one reason: loading the model *is*
the cold start. A 27B set of weights takes minutes to reach the GPU on a first run and ~190 s off a
warm network volume, so letting a request pick the model, the context length or a parser would make
every request a potential reload.

── The two parsers, and why one of them can be turned off ────────────────────────────────────────

`REASONING_PARSER` and `TOOL_CALL_PARSER` are what turn a raw completion into the `thinking` and
`tool_use` blocks a coding harness expects. They also interact badly, and the interaction is the
reason this worker exists at all:

  vllm-project/vllm#39056 — with `--reasoning-parser qwen3` plus a Qwen tool parser, a tool call the
  model emits *inside* `<think>` is swallowed. The reasoning parser takes everything before
  `</think>` as reasoning, so the tool parser never sees the markup, and the response comes back
  with populated reasoning and an empty `tool_calls`. Related: #19513 (reasoning breaks tool call
  parsing), #35221 (reasoning-only output lands in `content`), #19051 (400 on `tool_choice:
  required` plus reasoning).

The fix is not a patch — PR #39055 was closed unmerged, with the maintainer noting the Qwen3 parser
had been rewritten into a new streaming parser engine shipping in vLLM 0.24, which "specifically
handles extract tool calls from thinking sections". 0.25 generalised that into the Streaming Parser
Engine. **This image is built on a vLLM well past both**, which is the entire reason it is not
RunPod's `worker-v1-vllm`: that worker ships vLLM 0.23, and its own bump to 0.24 (PR #318) was
reverted (PR #321).

`REASONING_PARSER=""` is therefore a **supported mode, not a broken one**. With no reasoning parser
the `<think>` block stays in `content`, the tool parser sees the entire stream, and #39056 cannot
occur by construction. What is lost is the clean separation of thinking from answer, not the
thinking itself — the model reasons exactly as much either way. It is the escape hatch to reach for
if tool calls start disappearing after a base-image bump, and it is why `common.vllm_server.flags`
treats an empty value as "omit the flag" rather than passing `--reasoning-parser ""`, which vLLM
reads as a parser literally named `""` and dies on.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

# The model this endpoint was built for. Overridable because the image is not model-specific — the
# passthrough handler has no idea what is behind it — but changing it means a new network volume
# holding different weights, so it is a deployment decision and not a runtime one.
DEFAULT_MODEL_ID: Final = "Qwen/Qwen3.6-27B-FP8"


class ConfigError(ValueError):
    """The worker is configured in a way that cannot serve."""


# `environ` is threaded through every reader rather than reached for globally. It is the seam the
# test suite hangs off; a helper that quietly consulted `os.environ` would make `load_config({})`
# return whatever the machine running the tests happens to export.
def _int(environ: Mapping[str, str], name: str, default: int, minimum: int, maximum: int) -> int:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as err:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from err
    # Clamped rather than refused: a value out of range is a typo in a deployment manifest, and
    # refusing to start over one is a worse outcome than serving at the nearest legal setting.
    return min(maximum, max(minimum, value))


def _float(environ: Mapping[str, str], name: str, default: float, minimum: float, maximum: float) -> float:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as err:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from err
    return min(maximum, max(minimum, value))


def _bool(environ: Mapping[str, str], name: str, default: bool) -> bool:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{name} must be a boolean (1/0, true/false), got {raw!r}")


@dataclass(frozen=True, slots=True)
class Config:
    model_id: str
    """The Hugging Face repo actually loaded."""
    served_model_name: str
    """What clients must send as `model`. Defaults to `model_id`, so the two never silently differ."""
    max_model_len: int
    reasoning_parser: str
    """Empty means *no reasoning parser*. See the module docstring — this is a supported mode."""
    tool_call_parser: str
    enable_auto_tool_choice: bool
    enable_prefix_caching: bool
    gpu_memory_utilization: float
    tensor_parallel_size: int
    vllm_host: str
    vllm_port: int
    vllm_startup_timeout_s: int
    request_timeout_s: int
    """How long the handler waits on the local vLLM before giving up on one request."""
    max_concurrency: int
    """How many jobs RunPod may hand this worker at once. See the note in `handler.main`."""
    extra_args: tuple[str, ...]
    """Raw additional `vllm serve` arguments, whitespace-split. The escape hatch for a flag this
    config has no field for — a new engine option should not require an image rebuild to try."""


def load_config(env: Mapping[str, str] | None = None) -> Config:
    environ: Mapping[str, str] = os.environ if env is None else env

    model_id = environ.get("WORKER_MODEL_ID", "").strip() or DEFAULT_MODEL_ID
    # Defaulting to `model_id` rather than to a short alias is deliberate. Claude Code and every
    # OpenAI-shaped client send back exactly the string `/v1/models` advertised; a served name that
    # differs from the loaded repo means a 404 whose message names a model the operator never typed.
    served = environ.get("SERVED_MODEL_NAME", "").strip() or model_id

    enable_auto_tool_choice = _bool(environ, "ENABLE_AUTO_TOOL_CHOICE", True)
    tool_call_parser = environ.get("TOOL_CALL_PARSER", "qwen3_xml").strip()
    # vLLM refuses `--enable-auto-tool-choice` without a parser, and the error arrives minutes into
    # a cold start rather than at parse time. Catching it here turns a wasted GPU boot into a line
    # in the worker log before any weight is read.
    if enable_auto_tool_choice and tool_call_parser == "":
        raise ConfigError(
            "ENABLE_AUTO_TOOL_CHOICE is on but TOOL_CALL_PARSER is empty. vLLM requires a parser "
            "for automatic tool choice and fails at engine start-up, several minutes into a cold "
            "start. Set TOOL_CALL_PARSER, or turn ENABLE_AUTO_TOOL_CHOICE off."
        )

    return Config(
        model_id=model_id,
        served_model_name=served,
        max_model_len=_int(environ, "MAX_MODEL_LEN", 131_072, 1024, 1_048_576),
        reasoning_parser=environ.get("REASONING_PARSER", "qwen3").strip(),
        tool_call_parser=tool_call_parser,
        enable_auto_tool_choice=enable_auto_tool_choice,
        enable_prefix_caching=_bool(environ, "ENABLE_PREFIX_CACHING", True),
        # 0.92 rather than vLLM's own 0.90 default: FlashInfer's workspace and CUDA graph capture
        # sit outside the fraction vLLM reserves, and an OOM there surfaces as a cold start that
        # never completes rather than as an error naming memory.
        gpu_memory_utilization=_float(environ, "GPU_MEMORY_UTILIZATION", 0.92, 0.1, 0.99),
        # 1 on purpose: a 27B model quantised to FP8 is ~31 GB and fits one 80 GB card with room for
        # a 131k KV cache. Raising this splits one model across two GPUs and doubles the bill.
        tensor_parallel_size=_int(environ, "TENSOR_PARALLEL_SIZE", 1, 1, 8),
        vllm_host=environ.get("VLLM_HOST", "127.0.0.1"),
        vllm_port=_int(environ, "VLLM_PORT", 8000, 1, 65535),
        # Generous on purpose: this covers pulling ~31 GB off a network volume on the very first
        # run, plus engine init, torch.compile and CUDA graph capture — minutes each at this size.
        vllm_startup_timeout_s=_int(environ, "VLLM_STARTUP_TIMEOUT_S", 1800, 30, 7200),
        # Must stay under the endpoint's `execution_timeout_ms`. A long reasoning turn on a 27B
        # model genuinely runs for minutes; a tight value here truncates it into what a client
        # reports as a network error.
        request_timeout_s=_int(environ, "REQUEST_TIMEOUT_S", 900, 10, 14_400),
        # 🔴 The RunPod SDK hands a worker **one job at a time** unless a concurrency modifier says
        # otherwise, which would throw away vLLM's continuous batching entirely. 8 rather than
        # something larger because each in-flight request holds KV cache: past what the cache holds,
        # vLLM preempts and recomputes rather than refusing, which reads as latency with no error.
        max_concurrency=_int(environ, "MAX_CONCURRENCY", 8, 1, 256),
        extra_args=tuple(environ.get("VLLM_EXTRA_ARGS", "").split()),
    )
