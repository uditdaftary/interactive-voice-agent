"""Central Settings: defaults must equal the legacy hardcoded literals."""

import dataclasses

import pytest

from voice_agent.settings import Settings, load_settings


def test_defaults_match_legacy_literals() -> None:
    s = Settings()
    # D2: the LLM defaults moved from the old built-in assistant to the local Ollama `voice-agent` model.
    assert s.llm_base_url == "http://127.0.0.1:11434/v1"
    assert s.llm_model == "voice-agent"
    assert s.llm_api_key == "local"
    assert s.web_search_enabled is True
    assert s.inter_sentence_pause_s == 0.35
    assert s.followup_window_s == 2.5
    assert s.words_per_s == 2.5
    assert s.min_sentence_s == 1.2


def test_env_overrides() -> None:
    s = load_settings({"VOICE_AGENT_LLM_MODEL": "x", "VOICE_AGENT_WEB_SEARCH": "false"})
    assert s.llm_model == "x"
    assert s.web_search_enabled is False
    # Untouched fields keep their defaults.
    assert s.llm_base_url == Settings().llm_base_url


@pytest.mark.parametrize("raw", ["0", "false", "no", "FALSE", " No "])
def test_web_search_off_spellings(raw: str) -> None:
    assert load_settings({"VOICE_AGENT_WEB_SEARCH": raw}).web_search_enabled is False


@pytest.mark.parametrize("raw", ["1", "true", "yes", ""])
def test_web_search_other_values_stay_on(raw: str) -> None:
    assert load_settings({"VOICE_AGENT_WEB_SEARCH": raw}).web_search_enabled is True


def test_load_settings_with_empty_env_equals_defaults() -> None:
    assert load_settings({}) == Settings()


def test_repr_does_not_leak_the_api_key() -> None:
    settings = load_settings({"VOICE_AGENT_LLM_API_KEY": "sk-x"})
    assert settings.llm_api_key == "sk-x"
    assert "sk-x" not in repr(settings)
    assert "sk-x" not in str(settings)


def test_settings_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Settings().inter_sentence_pause_s = 1.0  # type: ignore[misc]  # frozen on purpose; the assignment is the test


def test_fast_zeroes_timings() -> None:
    s = Settings.fast()
    for name in (
        "inter_sentence_pause_s",
        "followup_window_s",
        "min_sentence_s",
    ):
        assert getattr(s, name) == 0.0, name
    # Rates are divisors, so "fast" means infinite rate: every derived duration is exactly 0.
    assert 7 / (s.words_per_s * 1.0) == 0.0
    # Non-timing fields are untouched.
    assert s.llm_model == Settings().llm_model
    assert s.web_search_enabled == Settings().web_search_enabled


def test_the_playback_ack_timeout_is_thirty_seconds_and_fast_leaves_it_alone() -> None:
    assert Settings().playback_ack_timeout_s == 30.0
    assert Settings.fast().playback_ack_timeout_s == 30.0  # a missing report must stall a test, not be waved through


def test_barge_in_listen_default_and_env_override() -> None:
    assert Settings().barge_in_listen_s == 8.0
    assert load_settings({"VOICE_AGENT_BARGE_IN_LISTEN_S": "3.5"}).barge_in_listen_s == 3.5
    assert Settings.fast().barge_in_listen_s == 8.0  # waiting for the user is not pacing; fast() leaves it alone


@pytest.mark.parametrize("raw", ["soon", "0", "-2", "nan", "inf", ""])
def test_barge_in_listen_rejects_nonsense_at_load(raw: str) -> None:
    with pytest.raises(ValueError, match="VOICE_AGENT_BARGE_IN_LISTEN_S"):
        load_settings({"VOICE_AGENT_BARGE_IN_LISTEN_S": raw})
