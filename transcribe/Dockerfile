# The RunPod serverless transcription worker: vLLM + Voxtral Small 24B, plus two bench challengers.
#
# Built on vLLM's own image rather than assembled from a CUDA base. vLLM's runtime is torch, a
# matching CUDA toolkit, FlashAttention and a long tail of kernels whose versions have to agree;
# reproducing that pairing here would be a second, worse copy of a thing upstream already publishes.
#
# ── The weights are NOT in this image, deliberately ──────────────────────────────────────────
# Voxtral Small is ~55 GB in bf16. Baking it in would mean pulling a ~60 GB image on every cold
# start of every worker, against reading the same bytes off a mounted network volume — which is
# strictly faster and is what `HF_HOME` below points at. The first run downloads ~48 GB into that
# volume once; the rollout does that deliberately, before any client is pointed at the endpoint.
#
# amd64 only. Every GPU this targets is x86, and the base image's aarch64 variant carries a
# different CUDA pairing that nothing here has been tested against.

# Pinned to a release tag, not `latest`: `latest` moved twice in the week this was written, and a
# silently-newer vLLM changes both the transcription API surface and the flags Voxtral needs.
FROM vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404

# ffmpeg is a hard dependency, not a convenience: every request is normalised to 16 kHz mono WAV
# before it reaches a model, which is what gives the length gate a duration to gate on and keeps
# the Ogg/Opus decode out of whichever libsndfile the image happens to carry. See worker/audio.py.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# `WITH_CHALLENGERS=1` by default so one image serves both production and the bench. The challenger
# backend is imported lazily inside its branch (transcribe/backends/__init__.py), so a production
# worker never loads librosa at all — it only carries it. Set to 0 for a lean image that cannot run
# `bench.py`'s model sweep.
ARG WITH_CHALLENGERS=1

# 🔴 **Every path here is relative to the REPO ROOT, not to this directory.** The build context is
# the root because `common/` is a sibling of `transcribe/`, and the workflow passes
# `context: .` with `file: transcribe/Dockerfile`. Writing `COPY requirements.txt` — which is what
# this file said while it lived at the root — fails with `"/requirements.txt": not found`, and it
# fails at build time rather than at run time, which is the good outcome.
COPY transcribe/requirements.txt transcribe/requirements-challengers.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_CHALLENGERS" = "1" ]; then pip install --no-cache-dir -r requirements-challengers.txt; fi

# `common` before `transcribe`: it changes far less often, so the layer it produces survives most
# rebuilds of the worker above it.
COPY common ./common
COPY transcribe ./transcribe
COPY bench.py ./

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface \
    WORKER_MODEL=voxtral-small-24b \
    WORKER_BIAS_MODE=none

# The base image's entrypoint starts vLLM's API server directly. This worker starts it itself —
# blocking, before the handler is registered, so FlashBoot snapshots a worker with the model
# already resident — so the entrypoint has to be cleared, not merely appended to.
ENTRYPOINT []
CMD ["python3", "-m", "transcribe.handler"]
