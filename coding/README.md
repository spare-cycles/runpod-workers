# coding worker

An OpenAI-route **passthrough** in front of vLLM, serving `Qwen/Qwen3.6-27B-FP8` on one A100 80 GB,
scale-to-zero. Built for a Claude Code–shaped client pointed at `ANTHROPIC_BASE_URL`, and generic
enough for any OpenAI or Anthropic SDK.

The endpoint — GPU type, network volume, worker ceiling, idle timeout — is declared in
[`Ephasme/iac-platform`](https://github.com/Ephasme/iac-platform) at `runpod/coding.yaml`.

## Why this image exists instead of `runpod/worker-v1-vllm`

That worker served this endpoint until 2026-08-03. Two of its properties could not be configured
around, and both were measured rather than inferred.

### 1. It ships vLLM 0.23, and the fix we need is in 0.24+

With `--reasoning-parser qwen3` plus a Qwen tool parser, a tool call the model emits *inside*
`<think>` is swallowed: the reasoning parser takes everything before `</think>` as reasoning, so the
tool parser never sees the markup and the response comes back with populated reasoning and an empty
`tool_calls` ([vllm#39056](https://github.com/vllm-project/vllm/issues/39056); related
[#19513](https://github.com/vllm-project/vllm/issues/19513),
[#35221](https://github.com/vllm-project/vllm/issues/35221),
[#19051](https://github.com/vllm-project/vllm/issues/19051)).

The fix is not a patch. [PR #39055](https://github.com/vllm-project/vllm/pull/39055) was **closed
unmerged**, the maintainer noting the Qwen3 parser had been rewritten into a new streaming parser
engine shipping in vLLM 0.24 that "specifically handles extract tool calls from thinking sections";
0.25 generalised it into the Streaming Parser Engine. RunPod's own bump to 0.24 (their PR #318) was
**reverted** (PR #321), so `worker-v1-vllm` cannot reach it.

This image is built on `vllm/vllm-openai:v0.26.0`, past both.

### 2. It proxies a closed set of routes and 500s on the rest

Measured on the live endpoint, 2026-08-03:

```
/v1/models                    HTTP 200
/v1/this-route-does-not-exist HTTP 500  {"status":500,"title":"Internal Server Error",…}
/v1/messages/count_tokens     HTTP 500  {"status":500,"title":"Internal Server Error",…}
```

`count_tokens` returns a response **byte-for-byte identical to a route that does not exist**. That
is not a gap in vLLM — its Python frontend serves the route — it is a route the worker never
forwarded. And the consequence is worse than a 404: the Anthropic SDK treats 5xx as *retryable*, so
the client hammered it ten times with backoff and stalled, where a 404 would have failed fast into
its own token estimate.

RunPod hands the worker `openai_route` and `openai_input` and lets **the worker** decide which routes
exist. So this one forwards all of them. New upstream routes work the day the base image ships them.

## The knobs that matter

| Env | Default | Why you would change it |
|---|---|---|
| `REASONING_PARSER` | `qwen3` | **Set to `""` to disable.** See below — this is the escape hatch, not a broken state. |
| `TOOL_CALL_PARSER` | `qwen3_xml` | Model-specific. Empty requires `ENABLE_AUTO_TOOL_CHOICE=0`. |
| `MAX_CONCURRENCY` | `8` | How many jobs RunPod may hand one worker at once. |
| `MAX_MODEL_LEN` | `131072` | Context. Costs KV cache. |
| `VLLM_EXTRA_ARGS` | — | Any engine flag this config has no field for. |

### `REASONING_PARSER=""` is a supported mode

With no reasoning parser the `<think>` block stays in `content`, the tool parser sees the entire
stream, and #39056 **cannot occur by construction**. What is lost is the clean separation of thinking
from answer, not the thinking itself — the model reasons exactly as much either way.

Reach for it if tool calls start disappearing after a base-image bump.

🔴 It has to mean *omit the flag*, not pass an empty one: vLLM reads `--reasoning-parser ""` as a
parser literally named `""` and dies at start-up. That is what
[`common.vllm_server.flags`](../common/vllm_server.py) exists for, and
`test_an_empty_reasoning_parser_omits_the_flag_entirely` is what keeps it true.

### Concurrency is not optional

🔴 The RunPod SDK hands a worker **one job at a time** unless a concurrency modifier says otherwise.
Measured against RunPod's own worker on 2026-08-03: three simultaneous 150-token requests on a single
worker returned in **4.0 s total**, against 3.8–4.3 s for the same request sent alone — vLLM's
continuous batching absorbs them for free. A coding harness fires the main turn, its background
calls and any subagent at once, so serialising them would surface as latency on every turn.

### Prefix caching can be defeated from the client

`--enable-prefix-caching` is on by default here, and it is the highest-leverage setting for a client
that re-sends a large, mostly identical system prompt every turn. But vLLM's own Claude Code guide
notes that Claude Code injects a **per-request hash** into the system prompt, which changes the
prefix every time. `CLAUDE_CODE_ATTRIBUTION_HEADER=0` on the client side is what stops that. A
cache-hit rate near zero in the vLLM logs is that, not a broken engine.

## The weights are never in the image

~31 GB of FP8 on a network volume, with `HF_HOME` and `VLLM_CACHE_ROOT` both pointing at it. Baking
them in would mean pulling a ~40 GB image on every cold start against reading the same bytes off a
mounted device. `runpod-sync.py` refuses to apply a spec whose `HF_HOME` is off-volume.
