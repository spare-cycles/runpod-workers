# The RunPod serverless coding worker: an OpenAI-route passthrough in front of vLLM.
#
# Built from the repo root, not from this directory — `common/` is a sibling. The workflow passes
# `context: .` and `file: coding/Dockerfile` for exactly that reason.
#
# ── Why this image exists at all ──────────────────────────────────────────────────────────────
# RunPod publishes `runpod/worker-v1-vllm`, and this endpoint ran on it until 2026-08-03. Two of its
# properties could not be configured around:
#
#   1. It ships **vLLM 0.23**. The fix for tool calls emitted inside `<think>` (vllm#39056) is not a
#      patch — PR #39055 was closed unmerged in favour of a parser rewrite that shipped in vLLM
#      0.24, generalised into the Streaming Parser Engine in 0.25. RunPod's own bump to 0.24 (their
#      PR #318) was reverted (PR #321), so that worker cannot reach the fix.
#   2. It proxies a **closed set of routes** — Chat Completions, Models, Responses, Messages —
#      and 500s on everything else, including `/v1/messages/count_tokens`. See `handler.py`.
#
# Both are gone by building on vLLM's own image and forwarding every route.
#
# amd64 only. Every GPU this targets is x86, and the base image's aarch64 variant carries a
# different CUDA pairing that nothing here has been tested against.

# Pinned to a release tag, not `latest`: `latest` moves, and a silently-newer vLLM changes both the
# parser behaviour this worker exists to get and the flags the engine accepts. Same base as the
# transcription worker in this repo, deliberately — one CUDA/torch pairing to reason about, and a
# bump gets exercised by two endpoints instead of one.
FROM vllm/vllm-openai:v0.26.0-x86_64-cu129-ubuntu2404

WORKDIR /app

COPY coding/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY coding ./coding

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    VLLM_CACHE_ROOT=/runpod-volume/vllm-cache

# The base image's entrypoint starts vLLM's API server directly. This worker starts it itself —
# blocking, before the handler is registered, so FlashBoot snapshots a worker with the model already
# resident — so the entrypoint has to be cleared, not merely appended to.
ENTRYPOINT []
CMD ["python3", "-m", "coding.handler"]
