"""Tests for configuration helpers."""

import pytest

from reachy_mini_conversation_app import config
from reachy_mini_conversation_app.config import (
    HF_BACKEND,
    OPENAI_GPT_LIVE_BACKEND,
    OPENAI_LIVE_DEFAULT_VOICE,
)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("45", 45.0),
        ("", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unset/blank falls back to the default
        ("soon", config.DEFAULT_APP_TIMEOUT_MINUTES),  # unparseable falls back to the default
        ("0", None),  # non-positive disables the watchdog
        ("-1", None),
    ],
)
def test_resolve_app_timeout_minutes(monkeypatch, raw_value, expected) -> None:
    """The env timeout parses to minutes, falls back to the default, or disables on non-positive."""
    monkeypatch.setenv(config.APP_TIMEOUT_MINUTES_ENV, raw_value)

    assert config.resolve_app_timeout_minutes() == expected


def test_conversation_backend_defaults_to_huggingface(monkeypatch) -> None:
    """Unset CONVERSATION_BACKEND keeps the Hugging Face path."""
    monkeypatch.delenv(config.CONVERSATION_BACKEND_ENV, raising=False)
    monkeypatch.setattr(config.config, "CONVERSATION_BACKEND", HF_BACKEND)

    config.refresh_runtime_config_from_env()

    assert config.get_conversation_backend() == HF_BACKEND
    assert config.get_default_voice() == config.HF_DEFAULTS.voice
    assert config.get_available_voices()[0] == config.HF_AVAILABLE_VOICES[0]


def test_conversation_backend_selects_openai_gpt_live(monkeypatch) -> None:
    """CONVERSATION_BACKEND=openai-gpt-live-1 switches voices and readiness to the Live path."""
    monkeypatch.setenv(config.CONVERSATION_BACKEND_ENV, OPENAI_GPT_LIVE_BACKEND)
    monkeypatch.delenv(config.OPENAI_API_KEY_ENV, raising=False)
    config.refresh_runtime_config_from_env()

    assert config.get_conversation_backend() == OPENAI_GPT_LIVE_BACKEND
    assert config.get_default_voice() == OPENAI_LIVE_DEFAULT_VOICE
    assert "marin" in config.get_available_voices()
    assert config.has_backend_configuration() is False

    monkeypatch.setenv(config.OPENAI_API_KEY_ENV, "sk-test")
    config.refresh_runtime_config_from_env()
    assert config.has_backend_configuration() is True


def test_invalid_conversation_backend_falls_back_to_huggingface(monkeypatch, caplog) -> None:
    """An unknown CONVERSATION_BACKEND value is ignored."""
    monkeypatch.setenv(config.CONVERSATION_BACKEND_ENV, "not-a-backend")

    with caplog.at_level("WARNING"):
        config.refresh_runtime_config_from_env()

    assert config.get_conversation_backend() == HF_BACKEND
    assert config.CONVERSATION_BACKEND_ENV in caplog.text
