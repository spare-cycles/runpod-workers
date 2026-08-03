"""The output-shape check that makes `chat` bias mode safe to try.

`chat` mode exists because it is the only path where context biasing definitely works: the model
is handed the audio *and* a text chunk naming the people in the conversation, and there is no doubt
that it reads both. The transcription endpoint has no such field — `mistral_common`'s
`TranscriptionRequest` carries no prompt at all — so biasing there is either vLLM's `hotwords`,
whose effect on Voxtral is unmeasured, or nothing.

The cost is that a language model asked to transcribe can do something else instead. Mistral's own
guidance is that transcription mode "maximizes performance" for transcription; asking the chat
endpoint moves the task into instruction-following territory, where the plausible failure is not
garbage but a *fluent summary* — which no error will ever surface, and which would be written into
a searchable transcript column as if it were speech.

Hence two blunt checks. Neither can tell a good transcript from a bad one; both reliably catch the
answer that is not a transcript at all. A rejection falls back to `none` for that request, so the
worst case is an unbiased transcript rather than a failed job.
"""

from __future__ import annotations

import re
from typing import Final

# Openers a model uses when it is narrating rather than transcribing. Matched at the *start* only —
# "voici" in the middle of a sentence is ordinary French, and matching it anywhere would reject real
# speech. Accents are matched explicitly rather than stripped: the input is already NFC from the
# model, and a normalisation pass here would be a second place for the two to disagree.
_META_OPENERS: Final = (
    r"voici (?:la |le |une |un )?(?:transcription|texte|retranscription)",
    r"here(?:'s| is) (?:the |a )?(?:transcription|transcript|text)",
    r"la transcription (?:de |du |est)",
    r"the (?:audio|recording|speaker) (?:says|is saying|contains)",
    r"l['’]?(?:audio|enregistrement) (?:dit|contient)",
    r"transcription\s*:",
    r"transcript\s*:",
    r"(?:je suis d[ée]sol[ée]|i['’]?m sorry|i cannot|i can['’]?t|je ne peux pas)",
    r"(?:r[ée]sum[ée]|summary)\s*:",
)

_META_RE: Final = re.compile(r"^\s*(?:" + "|".join(_META_OPENERS) + r")", re.IGNORECASE)

# Below this, the proportional floor is noise: a one-second "oui" is a legitimate voice note whose
# transcript is three characters, and no ratio can distinguish it from a refusal.
_MIN_FLOOR_CHARS: Final = 12


def min_chars_for(duration_s: float, chars_per_second: float) -> int:
    """The shortest output that could still plausibly be a transcript of `duration_s` of speech.

    French runs roughly 12-15 characters per second of speech, so the default 3.0 is about a fifth
    of that — deliberately far below the real rate. This is a floor for catching a *summary*, not an
    estimate of transcript length: set anywhere near the true rate it would start rejecting genuine
    recordings that are mostly silence, background noise, or a single word.
    """
    return max(_MIN_FLOOR_CHARS, int(duration_s * chars_per_second))


def reject_reason(text: str, duration_s: float, chars_per_second: float) -> str | None:
    """Why this is not a transcript, or None when it passes.

    Order matters: the meta check runs first so a short refusal is reported as a refusal rather than
    as "too short", which would send the reader looking at the audio instead of at the prompt.
    """
    stripped = text.strip()
    if stripped == "":
        return "the model returned nothing"
    if _META_RE.search(stripped) is not None:
        return f"the model answered with meta-commentary rather than a transcript: {stripped[:120]!r}"
    floor = min_chars_for(duration_s, chars_per_second)
    if len(stripped) < floor:
        return (
            f"the model returned {len(stripped)} characters for {duration_s:.1f}s of audio, "
            f"under the {floor}-character floor — this is a summary, not a transcript"
        )
    return None
