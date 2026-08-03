"""Starting vLLM inside the worker container, and waiting until it can actually answer.

**This runs before the handler is registered, and blocks.** That ordering is the whole point and is
not an accident of where the call happens to sit.

RunPod's FlashBoot snapshots a worker that has finished initialising and then restores it for later
cold starts. A worker whose model server starts lazily — on the first request, or in a thread the
handler polls — is snapshotted with nothing loaded, so *every* cold start pays the full model load
again and FlashBoot buys nothing. Blocking here costs one slow first boot per new worker and makes
every restored one warm.

── This module was `common/vllm_server.py` until 2026-08-04 ──────────────────────────────────────

The repo briefly held a second worker, and the lifecycle below — spawn, block on `/health`, tear
down — was the thing both of them shared. That worker was removed with the GPU endpoint it served
(`Ephasme/iac-platform`, `docs/runbooks/runpod-coding-endpoint.md`), and `common/`'s own rule was
that the bar for landing something there is a *second* worker already needing it. One caller left,
so it came back here rather than staying a longer import path.

The `ServerSpec` seam is kept even though there is now exactly one caller building exactly one argv.
It is what separates "how a vLLM is supervised" from "which flags this model needs", and that line
is the reusable part — the next worker starts by writing a `build_spec`, not by re-deriving the
FlashBoot constraint above. A `flags()` helper that turned an empty config value into an omitted
flag went with the coding worker: nothing here builds its argv as data, and a helper with no caller
is worse than no helper.
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
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from .config import Config

log: Final = logging.getLogger(__name__)

# How often to ask `/health` while waiting. Fast enough that a warm restore is not padded by the
# poll interval, slow enough not to spam a server that is busy loading weights.
_POLL_INTERVAL_S: Final = 2.0
_HEALTH_TIMEOUT_S: Final = 5.0
# How long a `SIGTERM` gets before the process is killed outright, at container shutdown.
_STOP_GRACE_S: Final = 20.0

# Where RunPod mounts a serverless network volume. Fixed by the platform: a template's own
# `volumeMountPath` is not honoured for serverless, so this is a constant and not a setting.
SERVERLESS_VOLUME_MOUNT: Final = "/runpod-volume"

_process: subprocess.Popen[bytes] | None = None


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """Everything the supervisor below needs to start one vLLM and know that it came up."""

    argv: tuple[str, ...]
    """The full command line. Built by `build_spec` — see the module docstring for why it is separate."""
    host: str
    port: int
    startup_timeout_s: int
    """How long to wait for `/health` before giving up on the worker entirely."""
    label: str
    """What to call this in the logs. The model id, in practice."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Extra environment for the child, layered over the container's own."""


def health_url(spec: ServerSpec) -> str:
    return f"http://{spec.host}:{spec.port}/health"


def base_url(spec: ServerSpec) -> str:
    """The root a client posts to. No `/v1` — callers append their own route."""
    return f"http://{spec.host}:{spec.port}"


def is_healthy(spec: ServerSpec) -> bool:
    """One `/health` probe. Never raises — every failure mode here is "not yet"."""
    try:
        with urllib.request.urlopen(health_url(spec), timeout=_HEALTH_TIMEOUT_S) as res:
            return 200 <= res.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def python_argv(*args: str) -> tuple[str, ...]:
    """`python -m vllm.entrypoints.openai.api_server ...`, the invocation this worker starts from.

    The module form rather than the `vllm` console script: the script is resolved off `PATH`, and
    the base image has more than one Python on it. `sys.executable` is the interpreter this worker
    is already running under, which is by construction the one vLLM was installed into.
    """
    return (sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args)


def build_spec(config: Config) -> ServerSpec:
    """This worker's vLLM command line — the part that is genuinely Voxtral's."""
    argv = python_argv(
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
    return ServerSpec(
        argv=argv,
        host=config.vllm_host,
        port=config.vllm_port,
        startup_timeout_s=config.vllm_startup_timeout_s,
        label=config.model_id,
    )


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


def start_spec(spec: ServerSpec) -> None:
    """Launch vLLM and block until it is serving, or raise.

    Raising is correct here rather than degrading: a worker with no model server cannot answer
    anything, and one that starts anyway would accept jobs and fail every one of them — which
    RunPod's scaler reads as throughput. Dying makes the failure visible in the worker logs on the
    first boot instead of in a client's error handler forever.
    """
    global _process  # noqa: PLW0603 — one server per container; a module-level handle is the honest shape.

    if is_healthy(spec):
        # A restored FlashBoot snapshot, or a hand-started server during development.
        log.info("vllm: already serving on port %s", spec.port)
        return

    log.info("vllm: starting %s (this is the cold start)", spec.label)
    env = dict(os.environ)
    env.update(spec.env)
    # Weights live on the network volume, never in the image: a ~60 GB image would have to be pulled
    # on every cold start, which is strictly worse than reading the same bytes off a mounted volume.
    env.setdefault("HF_HOME", f"{SERVERLESS_VOLUME_MOUNT}/huggingface")
    # `start_new_session` so a SIGINT delivered to the worker's process group does not race us to
    # the child: the atexit hook below is what stops it, in a defined order.
    _process = subprocess.Popen(list(spec.argv), env=env, start_new_session=True)
    atexit.register(_terminate)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sys.exit(0))

    deadline = time.monotonic() + spec.startup_timeout_s
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            raise RuntimeError(f"vllm exited with code {_process.returncode} before it began serving")
        if is_healthy(spec):
            log.info("vllm: serving %s", spec.label)
            return
        time.sleep(_POLL_INTERVAL_S)

    _terminate()
    raise RuntimeError(
        f"vllm did not answer {health_url(spec)} within {spec.startup_timeout_s}s. "
        "On a first run this usually means the weights are still downloading onto the network "
        "volume; raise VLLM_STARTUP_TIMEOUT_S or warm the volume once by hand."
    )


def start(config: Config) -> None:
    start_spec(build_spec(config))
