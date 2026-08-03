"""Exercises ffmpeg and ffprobe for real rather than stubbing them.

The failures this module exists to catch — a truncated base64 payload, a container with no
duration, an over-long recording — are all failures of the *interaction* with those binaries. A
mock would assert that the code calls ffmpeg the way the test author thinks ffmpeg works.
"""

import base64
import subprocess
from pathlib import Path

import pytest

from worker.audio import AudioError, decode_base64, prepare, probe_duration


def make_ogg(path: Path, seconds: float) -> bytes:
    """A real Ogg/Opus file of `seconds`, which is the container every WhatsApp voice note arrives in."""
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi"]
    argv += ["-i", f"sine=frequency=440:duration={seconds}"]
    argv += ["-ac", "1", "-c:a", "libopus", "-b:a", "16k", str(path)]
    subprocess.run(argv, check=True, capture_output=True)
    return path.read_bytes()


def test_decode_rejects_a_truncated_payload():
    """`validate=True` matters: Python's decoder otherwise discards stray characters silently.

    Without it a mangled payload decodes to plausible garbage and only fails much later, inside
    ffmpeg, as an unintelligible container error.
    """
    with pytest.raises(AudioError, match="not valid base64"):
        decode_base64("not base64 at all!!")


def test_decode_accepts_a_data_uri():
    payload = base64.b64encode(b"hello").decode("ascii")
    assert decode_base64(f"data:audio/ogg;base64,{payload}") == b"hello"


def test_decode_rejects_an_empty_payload():
    with pytest.raises(AudioError, match="required"):
        decode_base64("")


def test_prepare_converts_ogg_opus_to_16k_wav_and_reports_a_duration(tmp_path: Path):
    data = make_ogg(tmp_path / "note.ogg", 2.0)
    prepared = prepare(base64.b64encode(data).decode("ascii"), "note.ogg", tmp_path, max_seconds=900)
    assert prepared.path.suffix == ".wav"
    assert 1.8 < prepared.duration_s < 2.3
    # The gate, the model input and the reported duration all read the same file.
    assert probe_duration(prepared.path) == pytest.approx(prepared.duration_s, abs=0.05)


def test_prepare_refuses_a_recording_over_the_limit_before_converting(tmp_path: Path):
    """The gate sits between the probe and the conversion — the last point where refusing is free."""
    data = make_ogg(tmp_path / "long.ogg", 3.0)
    with pytest.raises(AudioError, match="over the 1s limit"):
        prepare(base64.b64encode(data).decode("ascii"), "long.ogg", tmp_path, max_seconds=1)
    assert not (tmp_path / "input.16k.wav").exists()


def test_prepare_rejects_bytes_that_are_not_audio(tmp_path: Path):
    payload = base64.b64encode(b"this is not a media container").decode("ascii")
    with pytest.raises(AudioError):
        prepare(payload, "note.ogg", tmp_path, max_seconds=900)
