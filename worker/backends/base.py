"""What every backend has to provide, and the request/response shapes shared across them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Request:
    path: Path
    """A 16 kHz mono WAV, already length-gated. See `audio.prepare`."""
    duration_s: float
    language: str | None
    """`None` means let the model detect. See the language policy note in the README."""
    bias_terms: tuple[str, ...] = field(default=())


@dataclass(frozen=True, slots=True)
class Result:
    text: str
    language: str | None
    bias_mode: str
    """What the backend *actually* used, which is not always what was configured: `chat` falls back
    to `none` when its output-shape guard rejects the answer, and that has to be visible to whoever
    reads the bench results."""
    notes: tuple[str, ...] = field(default=())
    """Anything the caller should know that is not an error — a bias-mode fallback, most often."""


class Backend(Protocol):
    def transcribe(self, req: Request) -> Result: ...
