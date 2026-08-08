from __future__ import annotations

import pytest

from uzbek_speech_entities.api.app import (
    _normalized_confidence_threshold,
    _speech_analysis_normalization_enabled,
    _speech_rescue_enabled,
)
from uzbek_speech_entities.config import load_config


def test_speech_rescue_flag_uses_config_and_strict_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    monkeypatch.delenv("SPEECH_NER_RESCUE_ENABLED", raising=False)
    assert _speech_rescue_enabled(config) is True
    monkeypatch.setenv("SPEECH_NER_RESCUE_ENABLED", "off")
    assert _speech_rescue_enabled(config) is False
    monkeypatch.setenv("SPEECH_NER_RESCUE_ENABLED", "YES")
    assert _speech_rescue_enabled(config) is True
    monkeypatch.setenv("SPEECH_NER_RESCUE_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="common boolean"):
        _speech_rescue_enabled(config)


def test_analysis_normalization_settings_are_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    monkeypatch.delenv("SPEECH_NER_ANALYSIS_NORMALIZATION_ENABLED", raising=False)
    assert _speech_analysis_normalization_enabled(config) is True
    assert _normalized_confidence_threshold(config) == 0.70
    monkeypatch.setenv("SPEECH_NER_ANALYSIS_NORMALIZATION_ENABLED", "off")
    assert _speech_analysis_normalization_enabled(config) is False
    monkeypatch.setenv("SPEECH_NER_ANALYSIS_NORMALIZATION_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="common boolean"):
        _speech_analysis_normalization_enabled(config)
