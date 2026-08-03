"""Every knob the worker reads, resolved once at import.

Two things are configuration here rather than request fields, deliberately.

**The model** (`WORKER_MODEL`) is fixed for the life of a worker because loading it *is* the cold
start: a 55 GB set of weights takes tens of seconds to reach the GPU, and letting a request pick
would make every request a potential model swap. The challengers exist so `bench.py` can run the
same corpus through a differently-configured endpoint, not so production can switch mid-flight.

**The bias mode** (`WORKER_BIAS_MODE`) is configuration for the opposite reason: it is *unproven*.
`hotwords` is a parameter vLLM accepts and whose effect on Voxtral nobody has measured; `chat`
abandons the transcription path entirely for one where biasing definitely works but the model may
answer with a summary instead of a transcript. Pinning it per-endpoint is what makes the bench a
controlled comparison and what makes a bad winner one env var away from being undone.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

# ── Models ────────────────────────────────────────────────────────────────────────────────────
#
# The production model and one challenger, kept wired in because none of the published French WER
# numbers were measured on noisy phone Opus — which is the only audio this worker will ever see.
# `bench.py` is what settles that; these entries are what it has to compare.
#
# ⚰️ **`whisper-fr` (`bofenghuang/whisper-large-v3-french`) was removed 2026-08-03**, unbenched. It
# lost on paper on two of three published French benchmarks, and the corpus that would have ranked
# it for real — private voice notes with hand-written references — does not exist and was not going
# to. Carrying a second inference stack for a comparison nobody was going to run is the cost this
# removes. `git log -- worker/backends/whisper_fr.py` restores it if that judgement turns out wrong.
#
# 🔴 It was also the **only** backend where `hotwords` genuinely worked — faster-whisper takes it as
# a first-class argument, which is what made it the control for the biasing question. That question
# is answered (`chat` won, `hotwords` proved inert on Voxtral), so the control has no remaining job;
# but if `hotwords` is ever revisited, note that this repo no longer contains a working example.
VOXTRAL_SMALL: Final = "voxtral-small-24b"
QWEN3_ASR: Final = "qwen3-asr"

MODEL_IDS: Final[dict[str, str]] = {
    VOXTRAL_SMALL: "mistralai/Voxtral-Small-24B-2507",
    # Challenger only, and the one id here that has not been exercised: confirm it against the Hub
    # before the bench run rather than trusting this line. `WORKER_MODEL_ID` overrides it.
    QWEN3_ASR: "Qwen/Qwen3-ASR-1.7B",
}

# ── Bias modes ────────────────────────────────────────────────────────────────────────────────
BIAS_NONE: Final = "none"
BIAS_HOTWORDS: Final = "hotwords"
BIAS_CHAT: Final = "chat"
BIAS_MODES: Final = (BIAS_NONE, BIAS_HOTWORDS, BIAS_CHAT)

# Mistral's own cap on `context_bias`, applied here too so a client that builds one list for both
# backends cannot have it silently truncated by whichever it happens to reach.
MAX_BIAS_TERMS: Final = 100

# The response payload cap on RunPod is 20 MB and a transcript is text, so this is not about the
# wire: it bounds a `chat`-mode answer that has started generating and will not stop.
MAX_TRANSCRIPT_CHARS: Final = 200_000


class ConfigError(ValueError):
    """The worker is configured in a way that cannot produce a transcript."""


# `environ` is threaded through every reader rather than reached for globally. It is the seam the
# whole test suite hangs off, and a helper that quietly consulted `os.environ` instead would make
# `load_config({})` return whatever the machine running the tests happens to export.
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


@dataclass(frozen=True, slots=True)
class Config:
    model: str
    """One of `MODEL_IDS`' keys. What kind of backend to build."""
    model_id: str
    """The Hugging Face repo actually loaded."""
    bias_mode: str
    vllm_host: str
    vllm_port: int
    vllm_startup_timeout_s: int
    """How long to wait for vLLM to answer `/health` before giving up on the worker entirely."""
    max_audio_seconds: int
    """Refused before any GPU time is spent. Kept in step with the endpoint's `execution_timeout_ms`."""
    request_timeout_s: int
    chat_min_chars_per_second: float
    """`chat` mode's output-shape floor. See `guards.py` for why it is a floor and not a target."""
    tensor_parallel_size: int
    gpu_memory_utilization: float

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    @property
    def needs_vllm(self) -> bool:
        """Only Voxtral is served through vLLM; the challengers load in-process."""
        return self.model == VOXTRAL_SMALL


def load_config(env: Mapping[str, str] | None = None) -> Config:
    environ: Mapping[str, str] = os.environ if env is None else env
    model = environ.get("WORKER_MODEL", VOXTRAL_SMALL).strip() or VOXTRAL_SMALL
    if model not in MODEL_IDS:
        raise ConfigError(f"WORKER_MODEL must be one of {sorted(MODEL_IDS)}, got {model!r}")

    bias_mode = environ.get("WORKER_BIAS_MODE", BIAS_NONE).strip() or BIAS_NONE
    if bias_mode not in BIAS_MODES:
        raise ConfigError(f"WORKER_BIAS_MODE must be one of {list(BIAS_MODES)}, got {bias_mode!r}")
    # A challenger has no chat template that would make `chat` mode mean anything, and vLLM is not
    # even running for it. Failing here beats failing per-request with a confusing 404.
    if bias_mode == BIAS_CHAT and model != VOXTRAL_SMALL:
        raise ConfigError(f"WORKER_BIAS_MODE=chat is only implemented for {VOXTRAL_SMALL}, not {model!r}")

    return Config(
        model=model,
        model_id=environ.get("WORKER_MODEL_ID", "").strip() or MODEL_IDS[model],
        bias_mode=bias_mode,
        vllm_host=environ.get("VLLM_HOST", "127.0.0.1"),
        vllm_port=_int(environ, "VLLM_PORT", 8000, 1, 65535),
        # Generous on purpose: this covers pulling ~48 GB off a network volume on the very first
        # run, plus vLLM's engine init and CUDA graph capture, which alone are minutes at this size.
        vllm_startup_timeout_s=_int(environ, "VLLM_STARTUP_TIMEOUT_S", 1800, 30, 7200),
        max_audio_seconds=_int(environ, "MAX_AUDIO_SECONDS", 900, 1, 14_400),
        request_timeout_s=_int(environ, "REQUEST_TIMEOUT_S", 900, 10, 14_400),
        chat_min_chars_per_second=_float(environ, "CHAT_MIN_CHARS_PER_SECOND", 3.0, 0.0, 100.0),
        # 1 on purpose: Voxtral Small is ~55 GB in bf16, which fits one 80 GB card. The model card's
        # `--tensor-parallel-size 2` example targets smaller cards, and copying it here would double
        # the GPU bill for nothing.
        tensor_parallel_size=_int(environ, "TENSOR_PARALLEL_SIZE", 1, 1, 8),
        gpu_memory_utilization=_float(environ, "GPU_MEMORY_UTILIZATION", 0.92, 0.1, 0.99),
    )
