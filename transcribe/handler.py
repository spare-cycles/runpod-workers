"""The RunPod serverless entry point, and the request contract every client codes against.

```jsonc
// POST https://api.runpod.ai/v2/<endpoint-id>/runsync
{"input": {
  "audio_base64": "…",        // required
  "filename":     "note.ogg", // the extension is the format hint; optional
  "language":     "fr",       // null or absent = let the model detect
  "bias_terms":   ["Marie", "Thibault", "Grenoble"]  // ≤100; ignored when bias mode is `none`
}}
// → {"text": "…", "language": "fr", "duration_s": 12.4, "infer_s": 6.1,
//    "model": "mistralai/Voxtral-Small-24B-2507", "bias_mode": "none", "notes": []}
```

**There is no `audio_url` field, and adding one would be a dead end.** Nothing in this design hosts
a file: the only client today is an MCP server with no object store and no public URL, so a URL
field would document a path nobody can take. At WhatsApp's ~16 kbps Opus the 900 s ceiling is about
2.4 MB of base64, comfortably inside RunPod's 10 MB request limit, and anything that blows it is a
transcoded video the caller should be gating on length instead.

**Errors are returned, not raised.** RunPod turns an exception into a job failure whose message is
whatever reached the top of the stack; returning `{"error": …}` keeps the wording under this
module's control and keeps a bad request — the common case — from being indistinguishable from a
GPU that fell over.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Final

from . import audio, vllm_server
from .backends import Request, make_backend
from .config import MAX_BIAS_TERMS, Config, ConfigError, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log: Final = logging.getLogger("worker")

# Two-letter ISO 639-1, which is what both vLLM and the Mistral API expect. Anything longer is
# almost certainly a locale (`fr-FR`) or a language name, and passing it through produces a
# confusing 400 from the model server rather than a clear one from here.
_LANGUAGE_MAX_LEN: Final = 2


def _config() -> Config:
    return load_config()


def _parse_language(raw: Any) -> str | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("language must be a two-letter code or null")
    value = raw.strip().lower()
    if value == "":
        return None
    if len(value) != _LANGUAGE_MAX_LEN or not value.isalpha():
        raise ValueError(f"language must be a two-letter ISO 639-1 code (got {raw!r}); use null to auto-detect")
    return value


def _parse_bias_terms(raw: Any) -> tuple[str, ...]:
    """Deduplicated, order-preserving, capped.

    The cap is Mistral's `context_bias` limit rather than anything vLLM imposes, applied on both
    paths so a client that builds one list for both backends gets the same terms honoured by
    whichever it reaches — instead of a silent truncation on one side only.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError("bias_terms must be a list of strings")
    seen: dict[str, None] = {}
    for item in raw:
        if not isinstance(item, str):
            raise ValueError("bias_terms must contain only strings")
        term = item.strip()
        if term != "":
            seen.setdefault(term, None)
    return tuple(seen)[:MAX_BIAS_TERMS]


def transcribe(job_input: dict[str, Any], config: Config, backend: Any) -> dict[str, Any]:
    """The whole of one job, minus the RunPod envelope. Separated so a test can drive it directly."""
    language = _parse_language(job_input.get("language"))
    bias_terms = _parse_bias_terms(job_input.get("bias_terms"))
    filename = job_input.get("filename") or "audio.ogg"
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")

    with audio.scratch_dir() as scratch:
        prepared = audio.prepare(
            audio_base64=job_input.get("audio_base64"),
            filename=filename,
            scratch=Path(scratch),
            max_seconds=config.max_audio_seconds,
        )
        started = time.monotonic()
        result = backend.transcribe(
            Request(
                path=prepared.path,
                duration_s=prepared.duration_s,
                language=language,
                bias_terms=bias_terms,
            )
        )
        infer_s = time.monotonic() - started

    # Stripped before the emptiness test, not after: a backend that answers with whitespace has
    # found no speech, and storing " " as a transcript would make every later call read it as a
    # cache hit and never transcribe the recording again.
    text = result.text.strip()
    if text == "":
        raise ValueError("no speech was found in this recording")

    return {
        "text": text,
        "language": result.language,
        "duration_s": round(prepared.duration_s, 2),
        "infer_s": round(infer_s, 2),
        "model": config.model_id,
        "bias_mode": result.bias_mode,
        "notes": list(result.notes),
    }


def make_handler(config: Config, backend: Any):  # noqa: ANN201 — the RunPod SDK types this as a bare callable.
    def handler(job: dict[str, Any]) -> dict[str, Any]:
        job_input = job.get("input")
        if not isinstance(job_input, dict):
            return {"error": "the job carried no `input` object"}
        try:
            return transcribe(job_input, config, backend)
        except (ValueError, audio.AudioError) as err:
            # The caller's problem: a bad payload, an over-long recording, silence. Named as such so
            # a client can tell it apart from an endpoint that is broken.
            return {"error": f"{type(err).__name__}: {err}"}
        # Broad on purpose: a handler that raises takes the worker down with it, and RunPod's scaler
        # reads a dying worker as a broken endpoint rather than as one bad request.
        except Exception as err:
            log.exception("worker: transcription failed")
            return {"error": f"transcription failed: {type(err).__name__}: {err}"}

    return handler


def main() -> None:
    import runpod

    config = _config()
    log.info("worker: model=%s bias_mode=%s", config.model_id, config.bias_mode)
    # Before the handler is registered, and blocking — see `transcribe.vllm_server`. FlashBoot snapshots
    # a worker once it is idle, so a model that loads lazily is a model that is never in the snapshot.
    if config.needs_vllm:
        vllm_server.start(config)
    backend = make_backend(config)
    runpod.serverless.start({"handler": make_handler(config, backend)})


if __name__ == "__main__":
    try:
        main()
    except ConfigError as err:
        log.error("worker: %s", err)
        raise SystemExit(2) from err
