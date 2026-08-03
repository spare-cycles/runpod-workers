"""`bofenghuang/whisper-large-v3-french` through faster-whisper — a bench challenger, not production.

It is wired in for one reason: **none of the published French WER numbers were measured on noisy
phone Opus.** FLEURS is read news, MLS is audiobooks, Common Voice is crowd recordings in a quiet
room. Voxtral Small wins all three on paper (4.03 / 6.18 / 3.73 against this model's 4.84 / 7.28 /
3.98), and that is what the endpoint is configured for — but a 0.8-point gap on read speech is not
a promise about a voice note recorded in a car, and the only way to find out is to run both over
real recordings.

This backend also happens to be the one place `hotwords` is **documented and real**: faster-whisper
takes it as a first-class argument. That makes it a useful control in the bias bench — if biasing
helps here and not on Voxtral's `hotwords` path, the parameter is being ignored rather than the
idea being wrong.
"""

from __future__ import annotations

import re
from typing import Any, Final

from ..config import BIAS_CHAT, BIAS_HOTWORDS, BIAS_NONE, Config
from .base import Request, Result

# Anything whisper puts in square brackets: `[BLANK_AUDIO]`, `[MUSIQUE]`, and the
# `[00:00:00.000 --> …]` timestamps older builds emit. **Stripped here, in the only backend that
# produces them**, rather than in every client — Voxtral emits no such annotations, so a client-side
# regex would be a rule with no owner that can only damage legitimate bracketed speech.
#
# Excluding newlines from the span bounds the damage: a stray `[` in real speech costs its own line
# at worst, instead of swallowing every word up to the next bracket anywhere in the transcript.
_ANNOTATION_RE: Final = re.compile(r"\[[^\]\n]*\]")

_BEAM_SIZE: Final = 5


def clean(raw: str) -> str:
    """Strip whisper's non-speech annotations and collapse the line breaks it emits per segment."""
    return _ANNOTATION_RE.sub(" ", raw).replace("\n", " ").strip()


class WhisperFrBackend:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from faster_whisper import WhisperModel

            # float16 rather than int8: this is a quality challenger, and quantising it would make
            # the comparison against an unquantised Voxtral meaningless.
            self._model = WhisperModel(self._config.model_id, device="cuda", compute_type="float16")
        return self._model

    def transcribe(self, req: Request) -> Result:
        model = self._load()
        kwargs: dict[str, Any] = {"beam_size": _BEAM_SIZE, "language": req.language}
        mode = BIAS_NONE
        # `chat` is refused at config load for this model, so only `hotwords` can reach here.
        if self._config.bias_mode in (BIAS_HOTWORDS, BIAS_CHAT) and req.bias_terms:
            kwargs["hotwords"] = " ".join(req.bias_terms)
            mode = BIAS_HOTWORDS

        segments, info = model.transcribe(str(req.path), **kwargs)
        # `segments` is a generator: the transcription does not actually run until it is consumed.
        text = clean(" ".join(segment.text for segment in segments))
        detected = getattr(info, "language", None)
        return Result(
            text=text,
            language=detected if isinstance(detected, str) and detected != "" else req.language,
            bias_mode=mode,
        )
