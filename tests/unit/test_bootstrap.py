from __future__ import annotations

import importlib
import wave
from pathlib import Path

import pytest

from uzbek_speech_entities.config import load_config, project_root, resolve_project_path
from uzbek_speech_entities.constants import APPLICATION_LABELS
from uzbek_speech_entities.logging_config import diagnostic_transcript_logging_enabled


def test_app_config_loads_exact_phase_zero_contract() -> None:
    config = load_config()

    assert config.path == project_root() / "configs/app.yaml"
    assert config.section("stt")["model_id"] == "navai-uz/whisper-small-uzbek"
    assert config.section("stt")["fallback_model_id"] == "navai-uz/whisper-base-uzbek"
    assert config.section("ner")["model_to_application_labels"]["TEMPORAL"] == "DATE"
    assert tuple(config.section("ner")["visible_labels"]) == APPLICATION_LABELS


def test_config_values_are_immutable() -> None:
    config = load_config()

    with pytest.raises(TypeError):
        config.values["app"] = {}  # type: ignore[index]


def test_relative_paths_resolve_against_explicit_root(tmp_path: Path) -> None:
    assert resolve_project_path("nested/file.txt", root=tmp_path) == tmp_path / "nested/file.txt"


def test_diagnostic_transcript_logging_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIAGNOSTIC_TRANSCRIPT_LOGGING", raising=False)
    assert diagnostic_transcript_logging_enabled() is False
    monkeypatch.setenv("DIAGNOSTIC_TRANSCRIPT_LOGGING", "true")
    assert diagnostic_transcript_logging_enabled() is True


def test_future_modules_import_without_ml_dependencies() -> None:
    module_names = [
        "uzbek_speech_entities.audio.validation",
        "uzbek_speech_entities.normalization.runtime",
        "uzbek_speech_entities.stt.transformers_backend",
        "uzbek_speech_entities.ner.predictor",
        "uzbek_speech_entities.pipeline.analyzer",
        "uzbek_speech_entities.api.app",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name)


def test_sample_audio_fixture_is_valid_mono_16khz_wav() -> None:
    fixture_path = project_root() / "tests/fixtures/sample_audio.wav"

    with wave.open(str(fixture_path), "rb") as audio_file:
        assert audio_file.getnchannels() == 1
        assert audio_file.getframerate() == 16_000
        assert audio_file.getnframes() > 0
