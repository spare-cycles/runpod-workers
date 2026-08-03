"""Starting vLLM inside the worker container, and waiting until it can actually answer.

**This runs at module import, before the handler is registered.** That ordering is the whole point
and is not an accident of where the call happens to sit.

RunPod's FlashBoot snapshots a worker that has finished initialising and then restores it for later
cold starts. A worker whose model server starts lazily — on the first request, or in a thread the
handler polls — is snapshotted with nothing loaded, so *every* cold start pays the full model load
again and FlashBoot buys nothing. Blocking here costs one slow first boot per new worker and makes
every restored one warm.

The flags are the model card's, minus `--tensor-parallel-size 2`: that example targets cards
smaller than this endpoint's. Voxtral Small is ~55 GB in bf16 and fits a single 80 GB A100, so
copying the example would double the GPU bill to serve one model twice as awkwardly.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Final

from .config import Config

log: Final = logging.getLogger(__name__)

# How often to ask `/health` while waiting. Fast enough that a warm restore is not padded by the
# poll interval, slow enough not to spam a server that is busy loading weights.
_POLL_INTERVAL_S: Final = 2.0
_HEALTH_TIMEOUT_S: Final = 5.0
# How long a `SIGTERM` gets before the process is killed outright, at container shutdown.
_STOP_GRACE_S: Final = 20.0

_process: subprocess.Popen[bytes] | None = None


def build_argv(config: Config) -> list[str]:
    """The `vllm serve` command line, as a list so nothing is ever shell-quoted."""
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
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
    ]


def health_url(config: Config) -> str:
    return f"http://{config.vllm_host}:{config.vllm_port}/health"


def is_healthy(config: Config) -> bool:
    """One `/health` probe. Never raises — every failure mode here is "not yet"."""
    try:
        with urllib.request.urlopen(health_url(config), timeout=_HEALTH_TIMEOUT_S) as res:
            return 200 <= res.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _terminate() -> None:
    """Stop vLLM on the way out, so a killed container does not leave a GPU held."""
    proc = _process
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=_STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()


def start(config: Config) -> None:
    """Launch vLLM and block until it is serving, or raise.

    Raising is correct here rather than degrading: a worker with no model server cannot transcribe
    anything, and one that starts anyway would accept jobs and fail every one of them — which
    RunPod's scaler reads as throughput. Dying makes the failure visible in the worker logs on the
    first boot instead of in a client's error handler forever.
    """
    global _process  # noqa: PLW0603 — one server per container; a module-level handle is the honest shape.

    if is_healthy(config):
        # A restored FlashBoot snapshot, or a hand-started server during development.
        log.info("vllm: already serving on port %s", config.vllm_port)
        return

    argv = build_argv(config)
    log.info("vllm: starting %s (this is the cold start)", config.model_id)
    env = dict(os.environ)
    # Weights live on the network volume, never in the image: a ~60 GB image would have to be pulled
    # on every cold start, which is strictly worse than reading the same bytes off a mounted volume.
    env.setdefault("HF_HOME", "/runpod-volume/huggingface")
    # `start_new_session` so a SIGINT delivered to the worker's process group does not race us to
    # the child: the atexit hook below is what stops it, in a defined order.
    _process = subprocess.Popen(argv, env=env, start_new_session=True)
    atexit.register(_terminate)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sys.exit(0))

    deadline = time.monotonic() + config.vllm_startup_timeout_s
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            raise RuntimeError(f"vllm exited with code {_process.returncode} before it began serving")
        if is_healthy(config):
            log.info("vllm: serving %s", config.model_id)
            return
        time.sleep(_POLL_INTERVAL_S)

    _terminate()
    raise RuntimeError(
        f"vllm did not answer {health_url(config)} within {config.vllm_startup_timeout_s}s. "
        "On a first run this usually means the weights are still downloading onto the network "
        "volume; raise VLLM_STARTUP_TIMEOUT_S or warm the volume once by hand."
    )
