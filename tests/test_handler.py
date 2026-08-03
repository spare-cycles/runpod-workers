"""The request contract, driven end to end with a fake backend.

`worker.handler` is importable without vLLM, torch or the RunPod SDK on purpose — every heavy
import in this package is inside the branch that needs it — which is what lets the contract be
tested on a CPU runner instead of only on a rented GPU.
"""

import base64
import subprocess
from pathlib import Path

import pytest

from worker.backends.base import Request, Result
from worker.config import load_config
from worker.handler import make_handler, transcribe


class FakeBackend:
    def __init__(self, result: Result | None = None, err: Exception | None = None) -> None:
        self.result = result or Result(text="bonjour, ça va ?", language="fr", bias_mode="none")
        self.err = err
        self.seen: Request | None = None

    def transcribe(self, req: Request) -> Result:
        self.seen = req
        if self.err is not None:
            raise self.err
        return self.result


@pytest.fixture
def ogg(tmp_path: Path) -> str:
    path = tmp_path / "note.ogg"
    argv = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2"]
    argv += ["-ac", "1", "-c:a", "libopus", "-b:a", "16k", str(path)]
    subprocess.run(argv, check=True, capture_output=True)
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_a_complete_response_carries_every_contract_field(ogg: str):
    config = load_config({})
    backend = FakeBackend()
    out = transcribe({"audio_base64": ogg, "filename": "note.ogg", "language": "fr"}, config, backend)
    assert set(out) == {"text", "language", "duration_s", "infer_s", "model", "bias_mode", "notes"}
    assert out["text"] == "bonjour, ça va ?"
    assert out["model"] == "mistralai/Voxtral-Small-24B-2507"
    assert 1.8 < out["duration_s"] < 2.3


def test_bias_terms_are_deduplicated_order_preserved_and_capped(ogg: str):
    config = load_config({})
    backend = FakeBackend()
    terms = ["Marie", "Thibault", "Marie", "  ", "Grenoble", *[f"n{i}" for i in range(200)]]
    transcribe({"audio_base64": ogg, "bias_terms": terms}, config, backend)
    assert backend.seen is not None
    assert backend.seen.bias_terms[:3] == ("Marie", "Thibault", "Grenoble")
    # Mistral's own `context_bias` limit, applied on both paths so one client-built list is honoured
    # identically by whichever backend it reaches.
    assert len(backend.seen.bias_terms) == 100


def test_an_absent_language_means_detect_rather_than_french(ogg: str):
    """98 % French is not 100 %. Forcing `fr` would silently mangle the other 2 %."""
    config = load_config({})
    backend = FakeBackend()
    transcribe({"audio_base64": ogg}, config, backend)
    assert backend.seen is not None
    assert backend.seen.language is None


def test_a_locale_is_rejected_rather_than_forwarded(ogg: str):
    config = load_config({})
    with pytest.raises(ValueError, match="two-letter"):
        transcribe({"audio_base64": ogg, "language": "fr-FR"}, config, FakeBackend())


def test_silence_is_an_error_not_an_empty_transcript(ogg: str):
    config = load_config({})
    backend = FakeBackend(Result(text="   ", language="fr", bias_mode="none"))
    with pytest.raises(ValueError, match="no speech"):
        transcribe({"audio_base64": ogg}, config, backend)


def test_a_bias_mode_fallback_reaches_the_caller_as_a_note(ogg: str):
    """`chat` falling back to `none` changes what was measured, so it cannot be silent."""
    config = load_config({})
    backend = FakeBackend(Result(text="bonjour " * 20, language="fr", bias_mode="none", notes=("fell back",)))
    out = transcribe({"audio_base64": ogg}, config, backend)
    assert out["bias_mode"] == "none"
    assert out["notes"] == ["fell back"]


def test_a_bad_request_is_returned_as_an_error_not_raised(ogg: str):
    """A raised handler takes the worker down with it, and RunPod reads that as a broken endpoint."""
    handler = make_handler(load_config({}), FakeBackend())
    assert "error" in handler({"input": {"audio_base64": "!!!"}})
    assert "error" in handler({"input": {}})
    assert "error" in handler({})


def test_a_backend_failure_is_reported_without_taking_the_worker_down(ogg: str):
    handler = make_handler(load_config({}), FakeBackend(err=RuntimeError("cuda oom")))
    out = handler({"input": {"audio_base64": ogg}})
    assert "cuda oom" in out["error"]
