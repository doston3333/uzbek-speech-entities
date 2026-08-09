from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from fakes import FakeNERPredictor, FakeSTTService

from uzbek_speech_entities.audio.validation import AudioValidationConfig
from uzbek_speech_entities.ner.schemas import Entity
from uzbek_speech_entities.pipeline.analyzer import SpeechEntityAnalyzer, TextValidationError


def _audio_config() -> AudioValidationConfig:
    return AudioValidationConfig(
        target_sample_rate=16_000,
        mono=True,
        max_seconds=60,
        max_upload_bytes=1024,
        allowed_extensions=frozenset({"wav"}),
    )


def test_text_analysis_preserves_raw_input_and_valid_offsets() -> None:
    text = " Akmal  Toshkentga "
    normalized = "Akmal Toshkentga"
    ner = FakeNERPredictor((Entity(text="Toshkentga", label="LOC", start=6, end=16, score=0.9),))
    analyzer = SpeechEntityAnalyzer(
        stt_service=FakeSTTService(), ner_predictor=ner, audio_config=_audio_config()
    )
    result = analyzer.analyze_text(text)
    assert result.raw_transcript == text
    assert result.normalized_transcript == normalized
    assert result.timing.audio_preprocessing_ms == result.timing.stt_ms == 0
    assert result.models.stt_revision is None
    assert all(value >= 0 for value in result.timing.model_dump().values())


@pytest.mark.parametrize("text", ["", "   ", "x" * 20_001])
def test_text_validation_rejects_blank_and_oversized_input(text: str) -> None:
    analyzer = SpeechEntityAnalyzer(
        stt_service=FakeSTTService(), ner_predictor=FakeNERPredictor(), audio_config=_audio_config()
    )
    with pytest.raises(TextValidationError):
        analyzer.analyze_text(text)


def test_audio_flow_uses_prepared_path_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    temporary = tmp_path / "canonical.wav"
    temporary.write_bytes(b"audio")
    observed: list[Path] = []

    @contextmanager
    def fake_prepared_audio(_: Path, __: AudioValidationConfig):
        try:
            yield temporary
        finally:
            temporary.unlink(missing_ok=True)

    monkeypatch.setattr(
        "uzbek_speech_entities.pipeline.analyzer.prepared_audio", fake_prepared_audio
    )
    text = "Akmal Toshkentga"
    analyzer = SpeechEntityAnalyzer(
        stt_service=FakeSTTService(text),
        ner_predictor=FakeNERPredictor(
            (Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),)
        ),
        audio_config=_audio_config(),
    )
    result = analyzer.analyze_audio(tmp_path / "source.wav")
    observed.append(temporary)
    assert result.raw_transcript == text
    assert result.models.stt_revision == "0000000000000000000000000000000000000000"
    assert not observed[0].exists()
