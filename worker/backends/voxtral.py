"""Voxtral Small 24B through vLLM's OpenAI-compatible API — the production backend.

Three bias modes live here, and the difference between them is which vLLM endpoint is used:

| mode       | endpoint                    | biasing                                              |
|------------|-----------------------------|------------------------------------------------------|
| `none`     | `/v1/audio/transcriptions`  | none — the baseline every other mode is measured against |
| `hotwords` | `/v1/audio/transcriptions`  | vLLM's `hotwords` extra parameter                     |
| `chat`     | `/v1/chat/completions`      | a text chunk naming the participants, beside the audio |

`none` is the default and stays the default until `bench.py` says otherwise. The other two are
hypotheses: `hotwords` is a parameter vLLM accepts whose effect on Voxtral nobody has published,
and `chat` trades the model's dedicated transcription mode — which Mistral says maximizes
transcription performance — for a path where biasing certainly reaches the model. `chat`'s answers
go through `guards.reject_reason` and fall back to `none` when they are not transcripts.

Requests are built with `mistral_common` rather than by hand. Voxtral's audio payloads have a
specific shape on both endpoints, and the library that defines that shape is the one that should
serialise it: a hand-rolled `{"type": "audio_url", …}` is a guess that would break silently the
next time the format moved.
"""

from __future__ import annotations

import logging
from typing import Any, Final

from openai import OpenAI

from .. import guards
from ..config import BIAS_CHAT, BIAS_HOTWORDS, BIAS_NONE, MAX_TRANSCRIPT_CHARS, Config
from .base import Request, Result

log: Final = logging.getLogger(__name__)

# vLLM does not check the API key, but the OpenAI client refuses to construct without one.
_PLACEHOLDER_KEY: Final = "EMPTY"

# Quality over speed, in one number. Greedy decoding is what makes two runs over the same corpus
# comparable, which is the whole premise of the bench.
_TEMPERATURE: Final = 0.0

# Roughly four characters per token, plus headroom, so a long recording is never cut off mid-word by
# a budget meant only to stop a runaway generation. Only `chat` mode needs this; the transcription
# endpoint bounds itself.
_CHARS_PER_TOKEN: Final = 3
_MIN_CHAT_TOKENS: Final = 512


def _chat_prompt(bias_terms: tuple[str, ...]) -> str:
    """The instruction for `chat` mode, with the biasing terms in it.

    Written to leave as little room as possible for the failure this mode has: every clause is
    aimed at "return the words, nothing else". The guard still runs — an instruction is a request,
    not a constraint — but a prompt that merely said "transcribe this" would fail far more often.
    """
    base = (
        "Transcribe this recording word for word. Output only the transcript itself: no preamble, "
        "no summary, no commentary, no speaker labels, no timestamps, and no translation — keep the "
        "language that is spoken."
    )
    if not bias_terms:
        return base
    terms = ", ".join(bias_terms)
    return (
        f"{base} These names and words are likely to occur in this recording; spell them exactly "
        f"this way when you hear them: {terms}."
    )


def _language_of(response: object) -> str | None:
    """The language the server reported, when it reported one.

    vLLM's transcription response carries `language` for some models and omits it for others, and
    the chat endpoint never carries it at all. Absent is a legitimate answer here — the field is
    echoed to the client as `null` rather than guessed at — because a wrong language label on a
    stored transcript is worse than no label.
    """
    value = getattr(response, "language", None)
    return value if isinstance(value, str) and value != "" else None


class VoxtralBackend:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = OpenAI(
            base_url=config.vllm_base_url,
            api_key=_PLACEHOLDER_KEY,
            timeout=float(config.request_timeout_s),
            # vLLM is one process on localhost; a client-side retry of a transcription that is
            # already running would double the GPU cost of every transient blip.
            max_retries=0,
        )

    # ── transcription endpoint ────────────────────────────────────────────────────────────────

    def _transcribe_endpoint(self, req: Request, hotwords: tuple[str, ...]) -> Result:
        from mistral_common.audio import Audio, RawAudio
        from mistral_common.protocol.transcription.request import TranscriptionRequest

        audio = Audio.from_file(str(req.path), strict=False)
        payload = TranscriptionRequest(
            model=self._config.model_id,
            audio=RawAudio.from_audio(audio),
            language=req.language,
            temperature=_TEMPERATURE,
            # `to_openai` emits every field the protocol defines; `top_p` and `seed` are not part of
            # the OpenAI transcription surface and vLLM rejects the request outright with them on.
        ).to_openai(exclude=("top_p", "seed"))

        extra: dict[str, Any] = {}
        if hotwords:
            # Not a documented Voxtral feature — vLLM exposes `hotwords` on the transcription API and
            # whether it reaches this model is exactly what the bench is for. Sent through
            # `extra_body` because the OpenAI SDK's typed signature has no such argument.
            extra["extra_body"] = {"hotwords": list(hotwords)}

        response = self._client.audio.transcriptions.create(**payload, **extra)
        text = (response.text or "").strip()
        mode = BIAS_HOTWORDS if hotwords else BIAS_NONE
        return Result(text=text, language=_language_of(response) or req.language, bias_mode=mode)

    # ── chat endpoint ─────────────────────────────────────────────────────────────────────────

    def _chat_endpoint(self, req: Request) -> Result:
        from mistral_common.audio import Audio
        from mistral_common.protocol.instruct.messages import AudioChunk, TextChunk, UserMessage

        audio = Audio.from_file(str(req.path), strict=False)
        message = UserMessage(
            content=[AudioChunk.from_audio(audio), TextChunk(text=_chat_prompt(req.bias_terms))]
        ).to_openai()

        max_tokens = max(_MIN_CHAT_TOKENS, int(req.duration_s * 20 / _CHARS_PER_TOKEN) + _MIN_CHAT_TOKENS)
        response = self._client.chat.completions.create(
            model=self._config.model_id,
            messages=[message],
            temperature=_TEMPERATURE,
            max_tokens=min(max_tokens, MAX_TRANSCRIPT_CHARS // _CHARS_PER_TOKEN),
        )
        text = (response.choices[0].message.content or "").strip()

        reason = guards.reject_reason(text, req.duration_s, self._config.chat_min_chars_per_second)
        if reason is None:
            return Result(text=text, language=req.language, bias_mode=BIAS_CHAT)

        # Falling back rather than failing: an unbiased transcript is a worse transcript, not a
        # broken one, and a job that dies here would cost the same GPU seconds and return nothing.
        log.warning("voxtral: chat-mode answer rejected (%s); falling back to the transcription endpoint", reason)
        fallback = self._transcribe_endpoint(req, hotwords=())
        return Result(
            text=fallback.text,
            language=fallback.language,
            bias_mode=BIAS_NONE,
            notes=(f"chat bias mode was rejected and fell back to none: {reason}",),
        )

    # ── entry point ───────────────────────────────────────────────────────────────────────────

    def transcribe(self, req: Request) -> Result:
        mode = self._config.bias_mode
        if mode == BIAS_CHAT:
            return self._chat_endpoint(req)
        hotwords = req.bias_terms if mode == BIAS_HOTWORDS else ()
        return self._transcribe_endpoint(req, hotwords=hotwords)
