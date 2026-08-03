"""Backend selection, kept to one function so a new model is one branch and one import.

Imports are **inside** the branch on purpose. `faster_whisper` and `transformers` are only present
for the challengers, and a top-level import of either would make a production worker — which loads
neither — pay their import cost on every cold start, and die outright if the image were ever
slimmed down to the production path alone.
"""

from __future__ import annotations

from ..config import QWEN3_ASR, VOXTRAL_SMALL, WHISPER_FR, Config, ConfigError
from .base import Backend, Request, Result

__all__ = ["Backend", "Request", "Result", "make_backend"]


def make_backend(config: Config) -> Backend:
    if config.model == VOXTRAL_SMALL:
        from .voxtral import VoxtralBackend

        return VoxtralBackend(config)
    if config.model == WHISPER_FR:
        from .whisper_fr import WhisperFrBackend

        return WhisperFrBackend(config)
    if config.model == QWEN3_ASR:
        from .qwen3 import Qwen3AsrBackend

        return Qwen3AsrBackend(config)
    # Unreachable: `load_config` refuses an unknown model. Kept so a new entry in `MODEL_IDS` that
    # forgets a branch here fails loudly instead of returning None.
    raise ConfigError(f"no backend is implemented for WORKER_MODEL={config.model!r}")
