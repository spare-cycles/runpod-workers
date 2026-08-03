"""Turning a base64 blob from a JSON request into a file a model can read.

Everything is normalised to **16 kHz mono WAV** before it reaches a backend, including WhatsApp's
Ogg/Opus, which is what almost every request will carry. That looks wasteful — Voxtral's own loader
can open an Ogg — but it buys three things that matter more than one ffmpeg process:

1. **One decoder, not several.** `mistral_common` reads audio through soundfile/libsndfile, whose
   Opus support depends on the version that happened to be built into the image. A silent decode
   failure inside the model server is far harder to diagnose than an ffmpeg exit code here.
2. **A duration before any GPU time is spent.** The length gate is the cheapest guard the worker
   has, and it can only run once something has parsed the container.
3. **A known sample rate.** Every model here wants 16 kHz mono; leaving the resample to whichever
   backend is loaded is how two backends end up disagreeing about what they were given.
"""

from __future__ import annotations

import base64
import binascii
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

FFMPEG: Final = "ffmpeg"
FFPROBE: Final = "ffprobe"

# A malformed container can make ffmpeg spin. These bound it; the request timeout is much larger and
# would let a wedged process hold a GPU worker for the whole execution budget.
FFMPEG_TIMEOUT_S: Final = 120
FFPROBE_TIMEOUT_S: Final = 30

# How much of a failed tool's stderr reaches the caller. Enough to diagnose, not enough to flood a
# JSON response that a client will put in front of a language model.
STDERR_TAIL_CHARS: Final = 600


class AudioError(ValueError):
    """The audio could not be decoded, probed or converted. Always the caller's problem to fix."""


@dataclass(frozen=True, slots=True)
class Prepared:
    path: Path
    """A 16 kHz mono WAV inside the scratch directory."""
    duration_s: float


def _tail(text: str) -> str:
    stripped = text.strip()
    return stripped if len(stripped) <= STDERR_TAIL_CHARS else f"…{stripped[-STDERR_TAIL_CHARS:]}"


def _run(argv: list[str], timeout_s: int) -> subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)
    except FileNotFoundError as err:
        raise AudioError(f"{argv[0]} is not installed in this image, and audio handling needs it") from err
    except subprocess.TimeoutExpired as err:
        raise AudioError(f"{argv[0]} did not finish within {timeout_s}s and was killed") from err
    if proc.returncode != 0:
        raise AudioError(f"{argv[0]} failed (exit {proc.returncode}): {_tail(proc.stderr.decode('utf8', 'replace'))}")
    return proc


def decode_base64(raw: str) -> bytes:
    """The request's `audio_base64`, as bytes.

    `validate=True` on purpose. Python's decoder silently *discards* characters outside the alphabet,
    so a truncated or double-encoded payload decodes to plausible-looking garbage that only fails
    much later, inside ffmpeg, as an unintelligible container error.
    """
    if not isinstance(raw, str) or raw == "":
        raise AudioError("audio_base64 is required and must be a non-empty string")
    # A data: URI is a common client mistake and costs one line to accept.
    payload = raw.split(",", 1)[1] if raw.startswith("data:") and "," in raw else raw
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as err:
        raise AudioError(f"audio_base64 is not valid base64: {err}") from err
    if len(data) == 0:
        raise AudioError("audio_base64 decoded to zero bytes")
    return data


def probe_duration(path: Path) -> float | None:
    """Seconds, or None when the container declares no duration (a raw stream, say)."""
    proc = _run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        FFPROBE_TIMEOUT_S,
    )
    text = proc.stdout.decode("utf8", "replace").strip()
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if seconds > 0 else None


def to_wav16k(src: Path, dest: Path) -> None:
    """Re-encode anything — a video's audio track included, hence `-vn` — as 16 kHz mono PCM."""
    argv = [FFMPEG, "-v", "error", "-y", "-i", str(src)]
    # `-vn` drops the video stream, so a video's audio track converts exactly like a voice note.
    argv += ["-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)]
    _run(argv, FFMPEG_TIMEOUT_S)


def prepare(audio_base64: str, filename: str, scratch: Path, max_seconds: int) -> Prepared:
    """Decode, probe, gate on length, and convert — in that order.

    The length gate sits between the probe and the conversion because that is the last point where
    refusing is free: after it, every step costs either CPU or GPU time on audio already known to be
    over budget.

    A container that declares **no** duration is let through rather than refused. Refusing would
    reject a legitimate raw stream on the strength of missing metadata; the request timeout is the
    backstop for the case where it really is hours long.
    """
    data = decode_base64(audio_base64)
    # The extension is the only format hint ffmpeg gets, and giving it the right one avoids a probe
    # that occasionally guesses wrong on a headerless stream. A filename with no extension is fine —
    # ffmpeg falls back to content sniffing.
    suffix = Path(filename or "audio.bin").suffix or ".bin"
    src = scratch / f"input{suffix}"
    src.write_bytes(data)

    duration = probe_duration(src)
    if duration is not None and duration > max_seconds:
        raise AudioError(
            f"this recording is {duration:.1f}s long, over the {max_seconds}s limit this endpoint accepts"
        )

    dest = scratch / "input.16k.wav"
    to_wav16k(src, dest)
    # The converted file is what every backend reads, so its duration is the one to report: a
    # container whose declared duration disagreed with its actual samples would otherwise make the
    # `chat`-mode length guard measure against a number nothing else saw.
    converted = probe_duration(dest)
    return Prepared(path=dest, duration_s=converted if converted is not None else (duration or 0.0))


def scratch_dir() -> tempfile.TemporaryDirectory[str]:
    """A per-request directory that cleans itself up, however the request ends.

    Serverless workers are reused across jobs, so a failure path that leaks a 10 MB WAV fills the
    container disk after a few hundred requests rather than never.
    """
    return tempfile.TemporaryDirectory(prefix="transcribe-")
