# runpod-workers

The RunPod serverless worker images for the fleet. One of them today, one Python package.

| Worker | Model | Shape | Image |
|---|---|---|---|
| [`transcribe/`](transcribe/README.md) | `mistralai/Voxtral-Small-24B-2507` | custom job (`audio_base64` → text) | `ghcr.io/spare-cycles/transcribe-worker` |

The endpoints themselves — GPU type, network volume, worker ceiling, idle timeout — are declared in
[`Ephasme/iac-platform`](https://github.com/Ephasme/iac-platform) under `runpod/` and reconciled by
`scripts/runpod-sync.py`. This repo is only the images that run on them.

## Why a plural name and one worker

This was `spare-cycles/transcribe-worker` until 2026-08-03, when a second worker — `coding/`, an
OpenAI-route passthrough in front of `Qwen/Qwen3.6-27B-FP8` — needed the same vLLM lifecycle this one
already had, and the repo was renamed to hold both.

That worker was removed on 2026-08-04, with the GPU endpoint it was built for. The endpoint was
deleted the day after it was created: it worked, but placing an A100 serverless worker on demand did
not, and a coding model you cannot reach is not a coding model. The full account — including the
finding that RunPod's only readable stock signal describes *pods* and not *serverless*, which cost an
afternoon twice — is in `iac-platform` at
[`docs/runbooks/runpod-coding-endpoint.md`](https://github.com/Ephasme/iac-platform/blob/main/docs/runbooks/runpod-coding-endpoint.md).

**The name stays.** Renaming back would move the GHCR package link a second time (see below) to buy
nothing, and `runpod-workers` is a correct name for a repo holding one RunPod worker.

The `common/` package went with the second worker, back into `transcribe/vllm_server.py`. Its own
rule was that the bar for landing something there is a *second* worker already needing it, and a
shared module with one caller is just a longer import path.

## The request shape is custom, not OpenAI

`transcribe` takes a custom job — base64 audio in, a transcript out — because there is no standard
API for "transcribe this with a bias term list and tell me what mode you used".

## Layout

```
transcribe/  Voxtral. config, audio pipeline, output guards, backends, vLLM lifecycle, Dockerfile.
tests/       one suite for the repo, run whole rather than per-directory.
bench.py     the transcription bench. Not part of the image's production path.
```

The Dockerfile builds **from the repo root**, not from its own directory, because `bench.py` is a
sibling. The workflow passes `context: .` and `file: transcribe/Dockerfile`.

## Local checks

```sh
uv venv .venv && uv pip install --python .venv/bin/python ruff pytest openai 'mistral-common[audio]'
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/python -m pytest -q
```

No vLLM, no torch, no GPU, no RunPod SDK. Every heavy import sits inside the branch that needs it,
which is what lets the request contract, the audio pipeline and the output guards be tested on a CPU
runner. `ffmpeg` is a real dependency of the audio tests — they build Ogg/Opus fixtures with it
rather than asserting that the code calls a mock the way its author imagined ffmpeg behaves.

## Releasing

The image is pinned **by digest** in `iac-platform`, never by tag — `runpod-sync.py` refuses a
mutable tag outright. So the deploy input is the digest each build prints as a workflow notice, and a
build that never publishes cannot move a running endpoint.

Release tags are `transcribe-v1.2.3`. A push to `main` publishes `:main` and `:sha-<short>`.

### The repository and its package have different names, on purpose

This repo is `runpod-workers`; it publishes `transcribe-worker`. That is not untidiness — it is what
made the rename safe, and it is still load-bearing.

A GHCR package links to its repository at **first publish**, by repository id, and there is no API to
change it afterwards. Creating a *new* repo would have left `ghcr.io/spare-cycles/transcribe-worker`
pointing at the old one, and the new repo's `GITHUB_TOKEN` would have been refused with
`403 Forbidden` on every push until someone fixed it by hand in the UI.

Renaming instead keeps the repository id (`1321244198` before and after, verified 2026-08-03), so the
package link, the Actions permissions and every old URL survive untouched. Renaming a repository does
not rename its packages, which is exactly the property being relied on: `runpod/transcribe.yaml` in
`iac-platform` goes on pinning the same package name by digest and never noticed.

🔴 **Consequence for CI:** `docker-transcribe.yml` must hard-code the image name. `ghcr.io/${{
github.repository }}` resolves to `ghcr.io/spare-cycles/runpod-workers`, a package nothing reads.

The rename was declared in `iac-platform` at `tofu/github-spare-cycles/main.tf` with a `moved` block,
removed once applied. 🔴 Editing that `for_each` key without one makes OpenTofu **delete the
repository** — issues, PRs and packages included — and create an empty one beside it, on a plan that
reads "1 to add, 1 to destroy" and looks exactly like a rename.
