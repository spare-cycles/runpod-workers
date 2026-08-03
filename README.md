# transcribe-worker

A RunPod serverless transcription endpoint: **`mistralai/Voxtral-Small-24B-2507`** on vLLM, bf16,
one A100 80 GB, scale-to-zero. Built for [`spare-cycles/whatsapp-mcp`](https://github.com/spare-cycles/whatsapp-mcp)
and deliberately generic enough for anything else that needs speech turned into text.

The endpoint itself — GPU type, network volume, worker ceiling, idle timeout — is declared in
[`Ephasme/iac-platform`](https://github.com/Ephasme/iac-platform) at `runpod/transcribe.yaml` and
reconciled by `scripts/runpod-sync.py`. This repo is only the image that runs on it.

## Why this model

French WER, from Mistral's Voxtral paper (arXiv 2507.13264, Tables 4-6 — one harness, so these
rows are directly comparable):

| Model | FLEURS-fr | CV-fr | MLS-fr |
|---|---|---|---|
| **Voxtral Small 24B** (Apache-2.0, ~55 GB bf16) | **4.03** | **6.18** | 3.73 |
| Voxtral Mini 3B (Apache-2.0) | 4.87 | 8.92 | 5.28 |
| Whisper large-v3 (baseline) | 5.55 | 11.33 | 5.09 |

It is the best French of anything self-hostable, and it edges even Mistral's own **closed** API
model on FLEURS-fr (4.03 against 4.32) — so self-hosting here is the *quality* choice, not merely
the private one. The workload is ~98 % French, which is why multilingual breadth (Qwen3-ASR's 52
languages) counts for nothing and why French-specific numbers are the only ones that matter.

**~55 GB in bf16 fits one 80 GB card.** The model card's `--tensor-parallel-size 2` example targets
smaller GPUs; copying it here would double the bill to serve one model twice as awkwardly.

⚠️ **None of those benchmarks is noisy phone Opus.** FLEURS is read news, MLS audiobooks, Common
Voice crowd recordings in a quiet room. That is why `Qwen3-ASR` stays wired in as a challenger and
why [`bench.py`](bench.py) exists.

⚰️ **`whisper-fr` was removed on 2026-08-03, unbenched.** It lost on paper on two of the three
published French benchmarks, and the corpus that would have ranked it for real — private voice notes
with hand-written references — does not exist and was not going to. A second inference stack
(`faster-whisper`, ctranslate2) carried in the image for a comparison nobody was going to run is
what that removal buys back. `git log -- worker/backends/whisper_fr.py` restores it.

## Context biasing is a question, not a feature

An early draft justified this model partly on "it takes a text prompt, so biasing is native". **That
does not hold on the transcription path**, and the correction shaped the whole design:

- `mistral_common`'s `TranscriptionRequest` exposes `id`, `model`, `audio`, `language`,
  `strict_audio_validation`, `streaming`, `target_streaming_delay_ms` — and **no prompt or bias
  field**.
- vLLM's transcription API does expose `hotwords`, but users report `prompt` has no observable
  effect for Voxtral or Whisper.
- The one place biasing is documented and real is the **Mistral API**'s `context_bias` — which is
  the *fallback* backend, and whose docs describe it as "optimized for English".

Biasing may still be worth a lot: proper nouns are where casual-speech WER accumulates, and a chat
client knows the participants' names. But it has to be **measured**, so the worker implements three
modes and the bench picked one.

✅ **Answered 2026-08-03 — `chat` won, `hotwords` is inert.** Ten French notes, one arm per mode:
5.24 % mean WER for `chat` against 17.36 % for both `none` and `hotwords`, with the name under test
going 0/10 to 9/10. `hotwords` returned **byte-identical text to the baseline on all ten** — at
`temperature=0` that means the parameter never reached the sampler, so vLLM accepts it and discards
it. Full result, and the paraphrase risk `chat` carries, in [`bench/README.md`](bench/README.md).

| `WORKER_BIAS_MODE` | Endpoint | Status |
|---|---|---|
| `none` *(default)* | `/v1/audio/transcriptions`, `temperature=0.0` | the baseline |
| `hotwords` | same, plus vLLM's `hotwords` list | exists in the API, **unverified for Voxtral** |
| `chat` | `/v1/chat/completions`, audio chunk + a text chunk naming the participants | biasing definitely reaches the model; **the model may summarize instead of transcribing** |

`chat` mode's answer goes through [`worker/guards.py`](worker/guards.py) — rejected if it is shorter
than a duration-proportional floor or opens with meta-commentary — and falls back to `none` for that
request. That guard is the only thing standing between a fluent summary and a searchable transcript
column holding it as if it were speech.

## The request contract

Stable. This is what every client codes against.

```jsonc
// POST https://api.runpod.ai/v2/<endpoint-id>/runsync
{"input": {
  "audio_base64": "…",        // required
  "filename":     "note.ogg", // the extension is the format hint; optional
  "language":     "fr",       // null or absent = let the model detect
  "bias_terms":   ["Marie", "Thibault", "Grenoble"]  // ≤100; ignored when bias mode is `none`
}}
// → {"text": "…", "language": "fr", "duration_s": 12.4, "infer_s": 6.1,
//    "model": "mistralai/Voxtral-Small-24B-2507", "bias_mode": "none", "notes": []}
```

A job that fails for the caller's reasons — bad base64, an over-long recording, silence — returns
`{"error": "…"}` with the class name in front. The handler never raises: an exception takes the
worker down, and RunPod reads a dying worker as a broken endpoint rather than a bad request.

**There is no `audio_url` field, and adding one would be a dead end.** Nothing in this design hosts
a file. At WhatsApp's ~16 kbps Opus the 900 s ceiling is about 2.4 MB of base64, comfortably inside
RunPod's 10 MB request limit; a payload that blows it is a transcoded video the caller should have
gated on length.

⚠️ **Three hosts, two of them sharing a `/v2` prefix.** `api.runpod.ai/v2/<id>` takes **jobs** —
that is this contract. Management lives elsewhere: `rest.runpod.io/v1` (the surface whose field
names are documented) with `api.runpod.io/v2` as a beta alias serving the same paths. A job sent to
either management host returns a 401 that reads exactly like a bad key.

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `WORKER_MODEL` | `voxtral-small-24b` | also `qwen3-asr` — a challenger, for the bench |
| `WORKER_MODEL_ID` | *(per model)* | override the Hugging Face repo without changing the backend |
| `WORKER_BIAS_MODE` | `none` | `none` \| `hotwords` \| `chat` |
| `MAX_AUDIO_SECONDS` | `900` | refused before any GPU time is spent; keep in step with the endpoint's `execution_timeout_ms` |
| `CHAT_MIN_CHARS_PER_SECOND` | `3.0` | `chat` mode's output floor — French runs 12-15, so this is deliberately a fifth of it |
| `VLLM_STARTUP_TIMEOUT_S` | `1800` | covers the first-ever run pulling ~48 GB onto the network volume |
| `TENSOR_PARALLEL_SIZE` | `1` | one 80 GB card. See above. |
| `GPU_MEMORY_UTILIZATION` | `0.92` | |
| `HF_HOME` | `/runpod-volume/huggingface` | **the weights live on the network volume, never in the image** |

The model is fixed for the life of a worker because loading it *is* the cold start. The bias mode
is fixed per endpoint because that is what makes the bench a controlled comparison.

## Benching

One run measures one configuration; sweeping means re-applying the endpoint spec between runs,
which is exactly what the rollout does.

```sh
export RUNPOD_API_KEY=… MISTRAL_API_KEY=…
./bench.py --corpus ./corpus --endpoint "$ENDPOINT_ID" --label voxtral/none
./bench.py --corpus ./corpus --endpoint "$ENDPOINT_ID" --label voxtral/hotwords   # after re-applying
./bench.py --corpus ./corpus --mistral --label mistral-api                        # the fallback, as a control
./bench.py --report
```

The corpus is a directory of `<name>.<audio>` each beside a `<name>.txt` reference, and optionally a
`<name>.bias.txt` with one biasing term per line. ~20 real voice notes is enough to separate the
candidates; include a couple of English ones so the language-detection policy is exercised too.

WER keeps accents (`ou` and `où` are different words) and unifies the two apostrophes, so a model
emitting U+2019 is not charged for every contraction.

The first request of a run records **cold-start wall time**, which is otherwise a number nobody in
this design has. Only meaningful if the endpoint really was scaled to zero first — the script
records it, it cannot assert it.

## Development

```sh
pip install ruff pytest openai
ruff check . && ruff format --check . && pytest -q
```

The test suite runs on a CPU with no vLLM, no torch, no GPU and no RunPod SDK: every heavy import
sits inside the branch that needs it, so the request contract, the `chat`-mode guard, the audio
pipeline and the WER metric are all testable on a laptop. What it cannot cover is the model itself
— that is what `bench.py` is for.
