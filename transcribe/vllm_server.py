"""This worker's vLLM command line, and nothing else.

The lifecycle — spawn, block on `/health`, tear down — lives in `common.vllm_server`, along with the
FlashBoot reasoning for why it blocks. What is left here is the part that is genuinely Voxtral's:
the three `mistral` formats and an explicit dtype.
"""

from __future__ import annotations

from common import vllm_server as server

from .config import Config


def build_spec(config: Config) -> server.ServerSpec:
    argv = server.python_argv(
        "--model",
        config.model_id,
        # The three `mistral` formats are required together for Voxtral: the tokenizer, the config
        # and the weights all use Mistral's own layout, and mixing one HF-format reader into the set
        # fails at load with an error that names none of the three.
        "--tokenizer_mode",
        "mistral",
        "--config_format",
        "mistral",
        "--load_format",
        "mistral",
        "--host",
        config.vllm_host,
        "--port",
        str(config.vllm_port),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        # bf16 explicitly rather than `auto`: quality over size is the whole reason this endpoint
        # exists, and `auto` would silently pick fp16 on a card that reports no bf16 support.
        "--dtype",
        "bfloat16",
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
