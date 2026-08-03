"""The two ways the Voxtral request can be wrong without any code path noticing.

Both of these shipped, and both got past the whole existing suite, because `test_handler.py` drives
the contract through a *fake* backend: nothing in it ever imports `mistral_common` or looks at the
argument list of `client.audio.transcriptions.create`. The first real request on a rented A100 is a
bad place to discover an `ImportError`.

1. **Symbols move between mistral-common modules.** `Audio` and `RawAudio` were importable from
   `mistral_common.audio` in earlier releases; by 1.11 that module holds only mel-spectrogram
   helpers and a deprecation shim. The failure is an `ImportError` raised inside the request, so it
   costs a cold start to find out. `RawAudio` is gone from the backend entirely now — it is
   deprecated in 1.11 and removed in 1.13 — but `Audio`, `AudioChunk`, `TextChunk` and
   `UserMessage` are still imported by name and can move the same way.
2. **`to_openai()` emits fields the OpenAI SDK's signature does not accept.** That signature has no
   `**kwargs`, so an extra key is a `TypeError` raised client-side — never a server response, never
   a vLLM log line. `_EXCLUDED_FROM_OPENAI` is the fix, and the test below recomputes the exclusion
   from the installed SDK instead of trusting the constant, so the next field mistral-common adds
   fails here rather than in production.

These need `mistral-common[audio]` and `openai` — both pure-Python plus numpy/soundfile, none of
vLLM, torch or a GPU. The rest of the suite still runs without them, which is why every import here
is inside the tests and the module is skipped wholesale when they are absent.
"""

import inspect
import subprocess
from pathlib import Path

import pytest

mistral_common = pytest.importorskip("mistral_common", reason="mistral-common is not installed")


@pytest.fixture
def wav(tmp_path: Path) -> str:
    """A second of real audio. `Audio.from_file` decodes through soundfile, so it has to be real."""
    path = tmp_path / "tone.wav"
    # fmt: off
    argv = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-ac", "1", "-ar", "16000", str(path),
    ]
    # fmt: on
    subprocess.run(argv, check=True)
    return str(path)


def test_every_symbol_the_backend_imports_still_exists():
    """Exactly the imports `worker.backends.voxtral` performs, in the modules it names.

    Asserting on the import itself rather than on a call: an `ImportError` here names the symbol
    that moved, where a failure further down would only say the request did not build.
    """
    from mistral_common.protocol.instruct.messages import AudioChunk, TextChunk, UserMessage
    from mistral_common.protocol.transcription.request import Audio, TranscriptionRequest

    assert hasattr(Audio, "from_file")
    assert hasattr(Audio, "to_base64")
    assert hasattr(AudioChunk, "from_audio")
    for cls in (TranscriptionRequest, TextChunk, UserMessage):
        assert callable(cls)


def _payload(wav: str, exclude: tuple[str, ...]) -> dict:
    from mistral_common.protocol.transcription.request import Audio, TranscriptionRequest

    audio = Audio.from_file(wav, strict=False)
    return TranscriptionRequest(
        model="mistralai/Voxtral-Small-24B-2507",
        audio=audio.to_base64("wav"),
        language="fr",
        temperature=0.0,
    ).to_openai(exclude=exclude)


def _accepted_params() -> set[str]:
    openai = pytest.importorskip("openai", reason="openai is not installed")
    create = openai.OpenAI(api_key="test").audio.transcriptions.create
    signature = inspect.signature(create)
    assert not any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()), (
        "the SDK grew a **kwargs, so an unexpected field would no longer raise — this test's premise is gone"
    )
    return set(signature.parameters)


def test_the_built_payload_only_uses_arguments_the_sdk_accepts(wav: str):
    from worker.backends.voxtral import _EXCLUDED_FROM_OPENAI

    payload = _payload(wav, _EXCLUDED_FROM_OPENAI)
    unexpected = set(payload) - _accepted_params()
    assert not unexpected, (
        f"`to_openai` emitted {sorted(unexpected)}, which `transcriptions.create` does not accept. "
        f"Add them to _EXCLUDED_FROM_OPENAI in worker/backends/voxtral.py."
    )
    # The exclusion must not have eaten the request either.
    assert payload["file"] is not None
    assert payload["model"] == "mistralai/Voxtral-Small-24B-2507"
    assert payload["language"] == "fr"
    assert payload["temperature"] == 0.0


def test_the_exclusion_list_is_load_bearing_and_not_cargo_cult(wav: str):
    """Every name in `_EXCLUDED_FROM_OPENAI` is there because the SDK rejects it.

    Without this, a field that mistral-common later stops emitting would sit in the list forever,
    and a reader would have no way to tell a live exclusion from a dead one.
    """
    from worker.backends.voxtral import _EXCLUDED_FROM_OPENAI

    emitted = set(_payload(wav, ()))
    accepted = _accepted_params()
    for name in _EXCLUDED_FROM_OPENAI:
        if name in emitted:
            assert name not in accepted, f"{name!r} is excluded but the SDK accepts it — drop it from the list"


def test_a_chat_message_serialises_to_an_audio_chunk_beside_the_text(wav: str):
    from mistral_common.protocol.instruct.messages import AudioChunk, TextChunk, UserMessage
    from mistral_common.protocol.transcription.request import Audio

    audio = Audio.from_file(wav, strict=False)
    message = UserMessage(content=[AudioChunk.from_audio(audio), TextChunk(text="transcris")]).to_openai()

    assert message["role"] == "user"
    assert [chunk["type"] for chunk in message["content"]] == ["input_audio", "text"]
