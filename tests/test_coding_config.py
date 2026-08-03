"""The coding worker's configuration, and the argv it builds.

The argv assertions are the point of this file. Every flag here is one that has to be right the
first time: the cheapest way to find out that `--reasoning-parser` was passed an empty string is
not a five-minute cold start ending in a lookup error on a rented A100.
"""

from __future__ import annotations

import pytest

from coding.config import DEFAULT_MODEL_ID, ConfigError, load_config
from coding.vllm_server import build_spec


def test_defaults_are_the_endpoint_this_was_built_for():
    config = load_config({})
    assert config.model_id == DEFAULT_MODEL_ID
    assert config.served_model_name == DEFAULT_MODEL_ID
    assert config.max_model_len == 131_072
    assert config.reasoning_parser == "qwen3"
    assert config.tool_call_parser == "qwen3_xml"


def test_served_model_name_defaults_to_the_loaded_repo():
    """A served name that differs from the loaded repo is a 404 naming a model nobody typed."""
    config = load_config({"WORKER_MODEL_ID": "Qwen/Other-27B"})
    assert config.served_model_name == "Qwen/Other-27B"


def test_out_of_range_values_are_clamped_not_refused():
    """A typo in a deployment manifest should not stop an endpoint from serving."""
    assert load_config({"GPU_MEMORY_UTILIZATION": "5"}).gpu_memory_utilization == 0.99
    assert load_config({"MAX_CONCURRENCY": "0"}).max_concurrency == 1


@pytest.mark.parametrize("value", ["banana", "1.5.2"])
def test_non_numeric_values_are_refused(value):
    with pytest.raises(ConfigError):
        load_config({"MAX_MODEL_LEN": value})


def test_auto_tool_choice_without_a_parser_is_refused_up_front():
    """vLLM fails this at engine start-up — minutes into a cold start. Catch it before the GPU."""
    with pytest.raises(ConfigError, match="TOOL_CALL_PARSER"):
        load_config({"TOOL_CALL_PARSER": "", "ENABLE_AUTO_TOOL_CHOICE": "1"})


# ── The argv ──────────────────────────────────────────────────────────────────────────────────


def argv_of(env):
    return list(build_spec(load_config(env)).argv)


def test_argv_carries_the_parsers_and_the_context_length():
    argv = argv_of({})
    assert "--reasoning-parser" in argv
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    assert argv[argv.index("--tool-call-parser") + 1] == "qwen3_xml"
    assert argv[argv.index("--max-model-len") + 1] == "131072"
    assert "--enable-auto-tool-choice" in argv


def test_an_empty_reasoning_parser_omits_the_flag_entirely():
    """🔴 The escape hatch for vllm#39056.

    `--reasoning-parser ""` is not "no parser" to vLLM — it looks up a parser literally named `""`
    and dies at start-up. Turning reasoning parsing off has to mean leaving the flag out, which is
    what makes it a usable mitigation rather than a second way to break the worker.
    """
    argv = argv_of({"REASONING_PARSER": ""})
    assert "--reasoning-parser" not in argv
    # The tool parser must survive: it is the half that still has a job with reasoning parsing off.
    assert "--tool-call-parser" in argv


def test_auto_tool_choice_can_be_turned_off_together_with_its_parser():
    argv = argv_of({"TOOL_CALL_PARSER": "", "ENABLE_AUTO_TOOL_CHOICE": "0"})
    assert "--tool-call-parser" not in argv
    assert "--enable-auto-tool-choice" not in argv


def test_prefix_caching_states_itself_in_both_directions():
    assert "--enable-prefix-caching" in argv_of({})
    assert "--no-enable-prefix-caching" in argv_of({"ENABLE_PREFIX_CACHING": "0"})


def test_extra_args_are_appended_verbatim():
    """The escape hatch for an engine flag this config has no field for."""
    argv = argv_of({"VLLM_EXTRA_ARGS": "--swap-space 8"})
    assert argv[-2:] == ["--swap-space", "8"]


def test_the_server_is_addressed_where_the_handler_looks_for_it():
    config = load_config({"VLLM_PORT": "9001"})
    spec = build_spec(config)
    assert spec.port == 9001
    assert spec.argv[spec.argv.index("--port") + 1] == "9001"
