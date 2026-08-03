"""This worker's vLLM command line.

The lifecycle — spawn, block on `/health`, tear down — lives in `common.vllm_server`, along with the
FlashBoot reasoning for why it blocks before the handler is registered.
"""

from __future__ import annotations

from common import vllm_server as server

from .config import Config


def build_spec(config: Config) -> server.ServerSpec:
    """The `vllm serve` argv for a coding model, built as data so empty values drop their flag."""
    argv = server.python_argv(
        *server.flags(
            [
                ("--model", config.model_id),
                ("--served-model-name", config.served_model_name),
                ("--host", config.vllm_host),
                ("--port", str(config.vllm_port)),
                ("--max-model-len", str(config.max_model_len)),
                ("--tensor-parallel-size", str(config.tensor_parallel_size)),
                ("--gpu-memory-utilization", str(config.gpu_memory_utilization)),
                # Both of these are omitted entirely when their value is empty, which is the whole
                # reason `flags` exists — see `config.py` on why turning the reasoning parser off is
                # a supported mode and not a misconfiguration.
                ("--reasoning-parser", config.reasoning_parser),
                ("--tool-call-parser", config.tool_call_parser),
            ]
        ),
        *(("--enable-auto-tool-choice",) if config.enable_auto_tool_choice else ()),
        # vLLM enables prefix caching by default on recent versions, but the flag pair is stated
        # explicitly: this is the single highest-leverage setting for an agentic client, which
        # re-sends a large and mostly identical system prompt on every turn.
        #
        # ⚠️ It can be defeated from the client side without anything here changing. vLLM's own
        # Claude Code guide notes that Claude Code injects a per-request hash into the system
        # prompt, which makes the prefix differ every turn; `CLAUDE_CODE_ATTRIBUTION_HEADER=0` on
        # the client is what stops that. A cache-hit rate near zero in the vLLM logs with this flag
        # on is that, not a broken engine.
        *(("--enable-prefix-caching",) if config.enable_prefix_caching else ("--no-enable-prefix-caching",)),
        *config.extra_args,
    )
    return server.ServerSpec(
        argv=argv,
        host=config.vllm_host,
        port=config.vllm_port,
        startup_timeout_s=config.vllm_startup_timeout_s,
        label=config.model_id,
    )


def start(config: Config) -> None:
    server.start(build_spec(config))
