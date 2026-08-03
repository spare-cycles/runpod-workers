#!/usr/bin/env python3
"""Measure one endpoint configuration over a corpus of real voice notes, and compare runs.

**Why this exists.** Every French WER number that chose the production model was measured on read
speech — FLEURS is news, MLS is audiobooks, Common Voice is people reading sentences in a quiet
room. None of it is a voice note recorded in a car at 16 kbps Opus. Two of the three decisions this
endpoint rests on are therefore *hypotheses* until measured here:

1. **Which model.** Voxtral Small 24B wins all three published French benchmarks. Whether it still
   wins on this corpus is the question.
2. **Which bias mode.** `hotwords` is a vLLM parameter whose effect on Voxtral nobody has
   published, and `chat` gives up the dedicated transcription path for one where biasing certainly
   reaches the model but the answer may be a summary. `none` is the default until this says
   otherwise.

**One run measures one configuration.** The model and the bias mode are properties of the deployed
endpoint, not of a request, so sweeping them means re-applying `runpod/transcribe.yaml` between
runs — which is exactly what the rollout does. Each run appends to `--results`, and `--report`
prints the comparison across everything recorded so far.

```sh
# One variant, against the deployed endpoint:
./bench.py --corpus ./corpus --endpoint "$RUNPOD_TRANSCRIBE_ENDPOINT_ID" --label voxtral/none
# The Mistral API, as the fallback backend's control:
./bench.py --corpus ./corpus --mistral --label mistral-api/context_bias
# Everything measured so far, side by side:
./bench.py --report
```

The corpus is a directory of `<name>.ogg` (or `.wav`, `.m4a`, …) each beside a `<name>.txt`
reference transcript, and optionally a `<name>.bias.txt` holding one biasing term per line — the
participant names a real client would supply. ~20 recordings is enough to separate the candidates;
a handful should be English so the language-detection policy gets exercised too.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

AUDIO_SUFFIXES: Final = (".ogg", ".opus", ".wav", ".m4a", ".mp3", ".mp4", ".webm", ".flac")

RUNPOD_JOBS_HOST: Final = "https://api.runpod.ai/v2"
MISTRAL_URL: Final = "https://api.mistral.ai/v1/audio/transcriptions"
MISTRAL_MODEL: Final = "voxtral-mini-latest"

# A100 80 GB flex, in dollars per second. Used only for the $/min column; the authoritative figure
# is always the RunPod console's, and this is here so a bench run can rank candidates by cost at all.
DEFAULT_PRICE_PER_SECOND: Final = 0.000756

REQUEST_TIMEOUT_S: Final = 1800
POLL_INTERVAL_S: Final = 3.0


# ── WER ───────────────────────────────────────────────────────────────────────────────────────

_PUNCT_RE: Final = re.compile(r"[^\w\s'’-]", re.UNICODE)
_SPACE_RE: Final = re.compile(r"\s+")


def normalise(text: str) -> list[str]:
    """Tokens for WER: case-folded, punctuation-stripped, accents **kept**.

    Stripping accents would be the usual convenience and is wrong here: `ou`/`où` and `a`/`à` are
    different words in French, and folding them makes a model that gets them wrong look identical
    to one that gets them right — on a corpus that is 98 % French, which is the entire point.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    # Apostrophes are word-internal in French (`l'audio`), so they survive the punctuation strip and
    # are then unified: a model that emits U+2019 must not be scored against a reference typed with
    # U+0027 as if every contraction were an error.
    folded = folded.replace("’", "'")
    return _SPACE_RE.sub(" ", _PUNCT_RE.sub(" ", folded)).strip().split()


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein over token lists, one row at a time.

    Rolled by hand rather than pulled from `jiwer` so the bench has no dependency beyond the
    standard library and the OpenAI client the worker already needs — it has to be runnable from a
    laptop against a deployed endpoint, not only inside the GPU image.
    """
    if not a:
        return len(b)
    previous = list(range(len(a) + 1))
    for j, bj in enumerate(b, start=1):
        current = [j]
        for i, ai in enumerate(a, start=1):
            current.append(min(previous[i] + 1, current[i - 1] + 1, previous[i - 1] + (ai != bj)))
        previous = current
    return previous[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


# ── transports ────────────────────────────────────────────────────────────────────────────────


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf8")
    request = urllib.request.Request(url, data=body, headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as res:
        return json.loads(res.read().decode("utf8"))


class RunPodTransport:
    """`/runsync`, falling back to polling `/status` when the job does not finish inline.

    ⚠️ Three hosts, two sharing a `/v2` prefix. **`api.runpod.ai/v2/<id>`** takes jobs — this
    class. Management is **`rest.runpod.io/v1`** (what `runpod-sync.py` in the iac-platform repo
    speaks), with **`api.runpod.io/v2`** as a beta alias onto the same paths. A job sent to either
    management host returns a 401 that reads exactly like a bad key.
    """

    def __init__(self, endpoint_id: str, api_key: str) -> None:
        self._base = f"{RUNPOD_JOBS_HOST}/{endpoint_id}"
        self._headers = {"authorization": f"Bearer {api_key}"}

    def transcribe(self, data: bytes, filename: str, language: str | None, bias: list[str]) -> dict[str, Any]:
        payload = {
            "input": {
                "audio_base64": base64.b64encode(data).decode("ascii"),
                "filename": filename,
                "language": language,
                "bias_terms": bias,
            }
        }
        body = _post_json(f"{self._base}/runsync", payload, self._headers)
        status = body.get("status")
        job_id = body.get("id")
        deadline = time.monotonic() + REQUEST_TIMEOUT_S
        while status in {"IN_QUEUE", "IN_PROGRESS"} and job_id is not None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"job {job_id} was still {status} after {REQUEST_TIMEOUT_S}s")
            time.sleep(POLL_INTERVAL_S)
            request = urllib.request.Request(f"{self._base}/status/{job_id}", headers=self._headers)
            with urllib.request.urlopen(request, timeout=60) as res:
                body = json.loads(res.read().decode("utf8"))
            status = body.get("status")
        if status != "COMPLETED":
            raise RuntimeError(f"job ended {status}: {body.get('error') or body}")
        output = body.get("output") or {}
        if "error" in output:
            raise RuntimeError(str(output["error"]))
        return output


class MistralTransport:
    """The fallback backend, as a control: `voxtral-mini-latest` with `context_bias`.

    Worth measuring even though it loses on paper (4.32 FLEURS-fr for the closed transcribe model
    against Voxtral Small's 4.03) because it is what every client falls back to when the endpoint is
    cold or down — so how much worse it is, on *this* corpus, is a number the runbook needs.

    Note that Mistral documents `context_bias` as optimized for English, on a corpus that is 98 %
    French. That is precisely why it is a measurement and not an assumption.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def transcribe(self, data: bytes, filename: str, language: str | None, bias: list[str]) -> dict[str, Any]:
        import io

        from openai import OpenAI

        client = OpenAI(base_url="https://api.mistral.ai/v1", api_key=self._api_key, max_retries=0)
        extra: dict[str, Any] = {}
        if bias:
            extra["extra_body"] = {"context_bias": bias}
        handle = io.BytesIO(data)
        handle.name = filename
        started = time.monotonic()
        response = client.audio.transcriptions.create(
            model=MISTRAL_MODEL,
            file=handle,
            # Deliberately not sending `timestamp_granularities`: Mistral rejects it together with
            # `language`, and language adherence matters more here than timestamps nobody stores.
            **({"language": language} if language else {}),
            **extra,
        )
        return {
            "text": response.text,
            "language": language,
            "infer_s": round(time.monotonic() - started, 2),
            "model": MISTRAL_MODEL,
            "bias_mode": "context_bias" if bias else "none",
        }


# ── the run ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class Sample:
    name: str
    wer: float
    duration_s: float
    infer_s: float
    wall_s: float
    language: str | None
    bias_terms: int
    text: str
    reference: str
    notes: list[str] = field(default_factory=list)


@dataclass
class Run:
    label: str
    model: str
    bias_mode: str
    samples: list[dict[str, Any]]
    mean_wer: float
    median_wer: float
    rtfx: float
    cold_start_s: float | None
    cost_per_audio_minute: float
    price_per_second: float


def load_corpus(directory: Path) -> list[tuple[Path, str, list[str]]]:
    """Every `<name>.<audio>` with a `<name>.txt` beside it, plus its optional biasing terms."""
    items: list[tuple[Path, str, list[str]]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        reference = path.with_suffix(".txt")
        if not reference.is_file():
            print(f"bench: skipping {path.name} — no {reference.name} beside it", file=sys.stderr)
            continue
        bias_file = path.with_suffix(".bias.txt")
        bias: list[str] = []
        if bias_file.is_file():
            bias = [line.strip() for line in bias_file.read_text("utf8").splitlines() if line.strip()]
        items.append((path, reference.read_text("utf8").strip(), bias))
    return items


def run_corpus(
    transport: Any,
    corpus: list[tuple[Path, str, list[str]]],
    language: str | None,
    use_bias: bool,
    price_per_second: float,
    label: str,
) -> Run:
    samples: list[Sample] = []
    cold_start_s: float | None = None
    model = "?"
    bias_mode = "?"

    for index, (path, reference, bias) in enumerate(corpus):
        data = path.read_bytes()
        started = time.monotonic()
        output = transport.transcribe(data, path.name, language, bias if use_bias else [])
        wall = time.monotonic() - started
        # The first request of a run is the only one that can be a cold start, and only if the
        # endpoint really was scaled to zero — which is the operator's job to arrange, not something
        # this script can assert. Recorded rather than claimed.
        if index == 0:
            cold_start_s = round(wall - float(output.get("infer_s") or 0.0), 2)
        model = str(output.get("model") or model)
        bias_mode = str(output.get("bias_mode") or bias_mode)
        text = str(output.get("text") or "")
        samples.append(
            Sample(
                name=path.name,
                wer=round(wer(reference, text), 4),
                duration_s=float(output.get("duration_s") or 0.0),
                infer_s=float(output.get("infer_s") or 0.0),
                wall_s=round(wall, 2),
                language=output.get("language"),
                bias_terms=len(bias) if use_bias else 0,
                text=text,
                reference=reference,
                notes=list(output.get("notes") or []),
            )
        )
        print(f"  {path.name}: WER {samples[-1].wer:.2%}  {samples[-1].infer_s:.1f}s", file=sys.stderr)

    wers = [s.wer for s in samples]
    total_audio = sum(s.duration_s for s in samples)
    total_infer = sum(s.infer_s for s in samples)
    return Run(
        label=label,
        model=model,
        bias_mode=bias_mode,
        samples=[asdict(s) for s in samples],
        mean_wer=round(statistics.fmean(wers), 4) if wers else 0.0,
        median_wer=round(statistics.median(wers), 4) if wers else 0.0,
        # Real time over inference time: how many seconds of audio one second of GPU handles.
        rtfx=round(total_audio / total_infer, 2) if total_infer > 0 else 0.0,
        cold_start_s=cold_start_s,
        # Charged on *wall* time, not inference time, because that is what RunPod bills — and the
        # idle tail on top of it is not visible from here at all. Under-counts on purpose in the one
        # direction that is safe: the ranking between candidates is what this column is for.
        cost_per_audio_minute=(
            round(sum(s.wall_s for s in samples) * price_per_second / (total_audio / 60), 4)
            if total_audio > 0
            else 0.0
        ),
        price_per_second=price_per_second,
    )


def report(results: Path) -> int:
    if not results.is_file():
        print(f"bench: no results at {results}", file=sys.stderr)
        return 1
    runs = [json.loads(line) for line in results.read_text("utf8").splitlines() if line.strip()]
    if not runs:
        print("bench: no runs recorded yet", file=sys.stderr)
        return 1
    width = max(len(r["label"]) for r in runs)
    header = f"{'label'.ljust(width)}  {'mean WER':>9}  {'median':>7}  {'RTFx':>6}"
    print(f"{header}  {'$/audio-min':>11}  {'cold s':>7}  n")
    for r in sorted(runs, key=lambda r: r["mean_wer"]):
        cold = "—" if r.get("cold_start_s") is None else f"{r['cold_start_s']:.0f}"
        print(
            f"{r['label'].ljust(width)}  {r['mean_wer']:>8.2%}  {r['median_wer']:>6.2%}  "
            f"{r['rtfx']:>6.2f}  {r['cost_per_audio_minute']:>11.4f}  {cold:>7}  {len(r['samples'])}"
        )
    best = min(runs, key=lambda r: r["mean_wer"])
    print(f"\nlowest mean WER: {best['label']}  (model={best['model']} bias_mode={best['bias_mode']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, help="directory of <name>.<audio> + <name>.txt pairs")
    parser.add_argument("--endpoint", help="RunPod endpoint id (uses RUNPOD_API_KEY)")
    parser.add_argument("--mistral", action="store_true", help="measure the Mistral API instead (MISTRAL_API_KEY)")
    parser.add_argument("--label", help="how this run appears in --report; defaults to the transport")
    parser.add_argument("--language", default=None, help="force a language code; omit to let the model detect")
    parser.add_argument("--no-bias", action="store_true", help="ignore every <name>.bias.txt")
    parser.add_argument("--price-per-second", type=float, default=DEFAULT_PRICE_PER_SECOND)
    parser.add_argument("--results", type=Path, default=Path("bench-results.jsonl"))
    parser.add_argument("--report", action="store_true", help="print every recorded run and exit")
    args = parser.parse_args(argv)

    if args.report:
        return report(args.results)
    if args.corpus is None:
        parser.error("--corpus is required unless --report is given")

    if args.mistral:
        key = os.environ.get("MISTRAL_API_KEY", "")
        if key == "":
            parser.error("MISTRAL_API_KEY is not set")
        transport: Any = MistralTransport(key)
        default_label = "mistral-api"
    elif args.endpoint:
        key = os.environ.get("RUNPOD_API_KEY", "")
        if key == "":
            parser.error("RUNPOD_API_KEY is not set")
        transport = RunPodTransport(args.endpoint, key)
        default_label = f"runpod/{args.endpoint}"
    else:
        parser.error("give either --endpoint or --mistral")

    corpus = load_corpus(args.corpus)
    if not corpus:
        parser.error(f"no usable <audio>/<name>.txt pairs in {args.corpus}")
    print(f"bench: {len(corpus)} recordings", file=sys.stderr)

    run = run_corpus(
        transport=transport,
        corpus=corpus,
        language=args.language,
        use_bias=not args.no_bias,
        price_per_second=args.price_per_second,
        label=args.label or default_label,
    )
    with args.results.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")
    print(
        f"\n{run.label}: mean WER {run.mean_wer:.2%}, median {run.median_wer:.2%}, "
        f"RTFx {run.rtfx}, cold start {run.cold_start_s}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as err:  # a 401 here is almost always the wrong host — see RunPodTransport
        print(f"bench: HTTP {err.code} — {err.read().decode('utf8', 'replace')[:400]}", file=sys.stderr)
        raise SystemExit(1) from err
