"""Starting vLLM inside a worker container, and waiting until it can actually answer.

**This runs before the handler is registered, and blocks.** That ordering is the whole point and is
not an accident of where the call happens to sit.

RunPod's FlashBoot snapshots a worker that has finished initialising and then restores it for later
cold starts. A worker whose model server starts lazily — on the first request, or in a thread the
handler polls — is snapshotted with nothing loaded, so *every* cold start pays the full model load
again and FlashBoot buys nothing. Blocking here costs one slow first boot per new worker and makes
every restored one warm.

── Why this module takes an argv and not a config ────────────────────────────────────────────────

It is shared by every worker in this repo, and what they share is the *lifecycle*: spawn, block on
`/health`, tear down on the way out. What they do **not** share is the flag list — Voxtral needs
`--tokenizer_mode mistral` and a bf16 dtype, a Qwen coding model needs a reasoning parser and a tool
parser, and the next one will need something else again. The first version of this file took the
transcription worker's `Config` object directly, which made every one of those flags reachable only
by adding a field to a dataclass that other workers would then carry and ignore.

So the seam is `ServerSpec`: the caller builds its own argv, and this module knows nothing about
what model is being served beyond a label for the log lines.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

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
    """Everything this module needs to start one vLLM and know that it came up."""

    argv: tuple[str, ...]
    """The full command line. Built by the caller — see the module docstring for why."""
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
    """`python -m vllm.entrypoints.openai.api_server ...`, the invocation every worker starts from.

    The module form rather than the `vllm` console script: the script is resolved off `PATH`, and
    the base image has more than one Python on it. `sys.executable` is the interpreter this worker
    is already running under, which is by construction the one vLLM was installed into.
    """
    return (sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args)


def flags(pairs: Sequence[tuple[str, str | None]]) -> tuple[str, ...]:
    """Flatten `[("--flag", "value"), ("--switch", None)]` into an argv fragment.

    Exists so a worker can build its flag list as data — which is what lets a config value of the
    empty string mean *do not pass this flag at all*, rather than passing it with an empty value.
    vLLM reads `--reasoning-parser ""` as a parser literally named `""` and dies at startup with a
    lookup error; omitting the flag is the only way to say "no parser". That distinction is load
    bearing for the coding worker, where turning the reasoning parser off is a supported mode.
    """
    out: list[str] = []
    for name, value in pairs:
        if value is None:
            out.append(name)
        elif value != "":
            out.extend((name, value))
    return tuple(out)


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


def start(spec: ServerSpec) -> None:
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
