# runpod-workers

The RunPod serverless worker images for the fleet. Two of them today, one Python package each, one
shared module.

| Worker | Model | Shape | Image |
|---|---|---|---|
| [`transcribe/`](transcribe/README.md) | `mistralai/Voxtral-Small-24B-2507` | custom job (`audio_base64` → text) | `ghcr.io/spare-cycles/transcribe-worker` |
| [`coding/`](coding/README.md) | `Qwen/Qwen3.6-27B-FP8` | OpenAI-route passthrough | `ghcr.io/spare-cycles/coding-worker` |

The endpoints themselves — GPU type, network volume, worker ceiling, idle timeout — are declared in
[`Ephasme/iac-platform`](https://github.com/Ephasme/iac-platform) under `runpod/` and reconciled by
`scripts/runpod-sync.py`. This repo is only the images that run on them.

## Why one repo

This was `spare-cycles/transcribe-worker` until the coding worker needed the same three things it
already had: a vLLM started *before* the handler is registered, a health-poll that blocks until the
engine really answers, and a teardown that does not leave a GPU held. That is
[`common/vllm_server.py`](common/vllm_server.py), and it is the whole of the shared surface.

Copying ~150 lines instead would have been defensible on size. It was rejected on drift: the
FlashBoot ordering that module encodes is exactly the kind of hard-won detail that gets fixed in one
copy and not the other, and this fleet has been bitten by spec/live drift more than once.

**The bar for landing something in `common/` is that a second worker already needs it.** A shared
module with one caller is just a longer import path, and it invites the next person to bend it to a
shape it was never designed for.

## Why the two workers do not share a request shape

`transcribe` takes a custom job — base64 audio in, a transcript out — because there is no standard
API for "transcribe this with a bias term list and tell me what mode you used".

`coding` is a passthrough: it forwards whatever OpenAI/Anthropic route it is handed to the local
vLLM and returns the answer verbatim. That is not laziness, it is the point — RunPod's own
`worker-v1-vllm` dispatches on a *closed* set of routes and returns a **500** for anything else,
including `/v1/messages/count_tokens`, which the Anthropic SDK then retries ten times because 5xx is
retryable. See [`coding/handler.py`](coding/handler.py) for the measurements.

## Layout

```
common/      shared by every worker. vllm_server: spawn, block on /health, tear down.
transcribe/  Voxtral. config, audio pipeline, output guards, backends, Dockerfile.
coding/      Qwen. config, route table, passthrough handler, Dockerfile.
tests/       one suite for the repo — a change to common/ must be proved against both workers.
bench.py     the transcription bench. Not part of either image's production path.
```

Each Dockerfile builds **from the repo root**, not from its own directory, because `common/` is a
sibling. The workflows pass `context: .` and `file: <worker>/Dockerfile`.

## Local checks

```sh
uv venv .venv && uv pip install --python .venv/bin/python ruff pytest openai 'mistral-common[audio]'
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
```

No vLLM, no torch, no GPU, no RunPod SDK. Every heavy import in both packages sits inside the branch
that needs it, which is what lets the request contracts, the audio pipeline, the output guards and
the route table all be tested on a CPU runner. `ffmpeg` is a real dependency of the audio tests —
they build Ogg/Opus fixtures with it rather than asserting that the code calls a mock the way its
author imagined ffmpeg behaves.

## Releasing

Both images are pinned **by digest** in `iac-platform`, never by tag — `runpod-sync.py` refuses a
mutable tag outright. So the deploy input is the digest each build prints as a workflow notice, and
a build that never publishes cannot move a running endpoint.

Tags are per worker: `transcribe-v1.2.3`, `coding-v1.2.3`. A push to `main` publishes `:main` and
`:sha-<short>` for whichever worker's paths changed.

### The repository and its packages have different names, on purpose

This repo is `runpod-workers`; it publishes `transcribe-worker` and `coding-worker`. That is not
untidiness — it is what made the move safe.

A GHCR package links to its repository at **first publish**, by repository id, and there is no API
to change it afterwards. Creating a *new* repo for the monorepo would have left
`ghcr.io/spare-cycles/transcribe-worker` pointing at the old one, and the new repo's
`GITHUB_TOKEN` would have been refused with `403 Forbidden` on every push until someone fixed it by
hand in the UI.

Renaming instead keeps the repository id (`1321244198` before and after, verified 2026-08-03), so
the package link, the Actions permissions and every old URL survive untouched. Renaming a repository
does not rename its packages, which is exactly the property being relied on: `runpod/*.yaml` in
`iac-platform` go on pinning the same two package names by digest and never noticed.

The rename itself is declared in `iac-platform` at `tofu/github-spare-cycles/main.tf`, with a
`moved` block. 🔴 Editing that key without one makes OpenTofu **delete the repository** — issues,
PRs and packages included — and create an empty one beside it, on a plan that reads "1 to add,
1 to destroy" and looks exactly like a rename.
