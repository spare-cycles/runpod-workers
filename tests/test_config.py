import pytest

from worker.config import BIAS_CHAT, BIAS_HOTWORDS, MODEL_IDS, QWEN3_ASR, VOXTRAL_SMALL, ConfigError, load_config


def test_defaults_are_the_production_configuration():
    config = load_config({})
    assert config.model == VOXTRAL_SMALL
    assert config.model_id == MODEL_IDS[VOXTRAL_SMALL]
    # `none` until the bench says otherwise — the other two modes are hypotheses, not features.
    assert config.bias_mode == "none"
    assert config.needs_vllm is True
    # One card. The model card's TP=2 example targets smaller GPUs than this endpoint's.
    assert config.tensor_parallel_size == 1


def test_unknown_model_is_refused_at_load():
    with pytest.raises(ConfigError, match="WORKER_MODEL"):
        load_config({"WORKER_MODEL": "whisper-tiny"})


def test_unknown_bias_mode_is_refused_at_load():
    with pytest.raises(ConfigError, match="WORKER_BIAS_MODE"):
        load_config({"WORKER_BIAS_MODE": "prompt"})


def test_chat_bias_mode_is_refused_for_a_challenger():
    """`chat` needs vLLM's chat endpoint, which is not running for an in-process challenger.

    Failing at load rather than per-request: the alternative is a worker that boots fine and then
    answers every job with a 404 from a server that was never started.
    """
    with pytest.raises(ConfigError, match="chat"):
        load_config({"WORKER_MODEL": QWEN3_ASR, "WORKER_BIAS_MODE": BIAS_CHAT})


def test_hotwords_is_allowed_for_a_challenger():
    # `hotwords` stays loadable on a challenger even though the 2026-08-03 bench proved it inert on
    # Voxtral: the mode is a property of the request path, and refusing it here would make the
    # config layer encode a measurement rather than a constraint. `chat` above is refused for the
    # opposite reason — it cannot physically work without vLLM's chat endpoint.
    config = load_config({"WORKER_MODEL": QWEN3_ASR, "WORKER_BIAS_MODE": BIAS_HOTWORDS})
    assert config.bias_mode == BIAS_HOTWORDS
    assert config.needs_vllm is False


def test_model_id_can_be_overridden_without_changing_the_backend():
    config = load_config({"WORKER_MODEL": VOXTRAL_SMALL, "WORKER_MODEL_ID": "mistralai/Voxtral-Mini-3B-2507"})
    assert config.model == VOXTRAL_SMALL
    assert config.model_id == "mistralai/Voxtral-Mini-3B-2507"


def test_numeric_env_is_clamped_rather_than_rejected():
    config = load_config({"MAX_AUDIO_SECONDS": "99999999", "GPU_MEMORY_UTILIZATION": "3"})
    assert config.max_audio_seconds == 14_400
    assert config.gpu_memory_utilization == 0.99


def test_a_non_numeric_env_is_a_config_error():
    with pytest.raises(ConfigError, match="MAX_AUDIO_SECONDS"):
        load_config({"MAX_AUDIO_SECONDS": "nine hundred"})
