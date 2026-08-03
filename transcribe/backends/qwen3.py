"""Qwen3-ASR through a plain transformers pipeline — the second bench challenger.

Present for completeness rather than because it is expected to win: 4.75 FLEURS-fr / 8.56 CV-fr /
5.26 MLS-fr puts it behind both Voxtral Small and the French Whisper, and its 52-language breadth
— the thing it is actually built for — is worth nothing to a workload that is 98 % French. It earns
its place in the bench only because the benchmarks above are read speech and this corpus is not.

A generic `automatic-speech-recognition` pipeline, deliberately. Anything more specific would be a
guess about an architecture this repo has never loaded, and a bench challenger is not worth a
bespoke loader; if it turns out to win, that is the moment to write one.

⚠️ `MODEL_IDS[qwen3-asr]` has not been verified against the Hub. Confirm it, or set
`WORKER_MODEL_ID`, before the bench run.
"""

from __future__ import annotations

from typing import Any, Final

from ..config import BIAS_NONE, Config
from .base import Request, Result

# 30 s is the window every Whisper-family encoder sees at once. Longer audio has to be chunked, and
# the overlap is what stops a word landing exactly on a boundary from being lost.
_CHUNK_LENGTH_S: Final = 30
_STRIDE_LENGTH_S: Final = 5


class Qwen3AsrBackend:
    def __init__(self, config: Config) -> None:
        self._config = config
        self._pipe: Any = None

    def _load(self) -> Any:
        if self._pipe is None:
            import torch
            from transformers import pipeline

            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self._config.model_id,
                torch_dtype=torch.bfloat16,
                device="cuda",
                chunk_length_s=_CHUNK_LENGTH_S,
                stride_length_s=_STRIDE_LENGTH_S,
            )
        return self._pipe

    def transcribe(self, req: Request) -> Result:
        pipe = self._load()
        kwargs: dict[str, Any] = {}
        if req.language is not None:
            kwargs["generate_kwargs"] = {"language": req.language}
        output = pipe(str(req.path), **kwargs)
        text = output["text"] if isinstance(output, dict) else str(output)
        # No biasing path at all: nothing in this pipeline takes hotwords, and reporting `none` is
        # what keeps the bench honest about which rows were actually biased.
        return Result(text=text.strip(), language=req.language, bias_mode=BIAS_NONE)
