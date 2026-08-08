from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from fakes import FakeNERPredictor, FakeSTTService

from uzbek_speech_entities.audio.validation import AudioValidationConfig
from uzbek_speech_entities.ner.schemas import Entity
from uzbek_speech_entities.pipeline.analyzer import SpeechEntityAnalyzer


def _audio_config() -> AudioValidationConfig:
    return AudioValidationConfig(16_000, True, 60, 1024, frozenset({"wav"}))


def test_audio_rescue_returns_the_precision_fixture_without_a_second_model_call(
    monkeypatch, tmp_path: Path
) -> None:
    text = (
        "assalomu alaykum mening ismim rajabov doston am men oltinchi yu avgust kuni soati "
        "uchida yangi oʻzbekiston universitetida boʻlaman am men oʻn sakkiz yoshdaman am biz "
        "sardor bilan birga ashxobod parkida koʻrishamiz"
    )
    canonical = tmp_path / "canonical.wav"
    canonical.write_bytes(b"audio")

    @contextmanager
    def fake_prepared_audio(_: Path, __: AudioValidationConfig):
        yield canonical

    monkeypatch.setattr(
        "uzbek_speech_entities.pipeline.analyzer.prepared_audio", fake_prepared_audio
    )
    anchors = tuple(
        Entity(
            text=value,
            label=label,
            start=text.index(value),
            end=text.index(value) + len(value),
            score=0.9,
        )
        for value, label in (("oʻzbekiston", "ORG"), ("ashxobod", "LOC"))
    )
    predictor = FakeNERPredictor(anchors)
    analyzer = SpeechEntityAnalyzer(
        stt_service=FakeSTTService(text),
        ner_predictor=predictor,
        audio_config=_audio_config(),
        normalizer=lambda value: value,
        speech_rescue_enabled=True,
    )
    result = analyzer.analyze_audio(tmp_path / "input.wav")
    observed = [
        (entity.label, entity.start, entity.end, entity.text, entity.source)
        for entity in result.entities
    ]
    assert observed == [
        ("PER", 30, 44, "rajabov doston", "person_introduction"),
        ("DATE", 52, 88, "oltinchi yu avgust kuni soati uchida", "temporal_grammar"),
        ("ORG", 89, 121, "yangi oʻzbekiston universitetida", "model_boundary_expansion"),
        ("PER", 166, 172, "sardor", "person_relation"),
        ("LOC", 185, 201, "ashxobod parkida", "model_boundary_expansion"),
    ]
    assert len(predictor.inputs) == 1


def test_rescue_is_audio_only_and_can_be_disabled() -> None:
    text = "mening ismim doston"
    predictor = FakeNERPredictor()
    analyzer = SpeechEntityAnalyzer(
        stt_service=FakeSTTService(text),
        ner_predictor=predictor,
        audio_config=_audio_config(),
        normalizer=lambda value: value,
        speech_rescue_enabled=True,
    )
    assert analyzer.analyze_text(text).entities == ()

    disabled = SpeechEntityAnalyzer(
        stt_service=FakeSTTService(text),
        ner_predictor=FakeNERPredictor(),
        audio_config=_audio_config(),
        normalizer=lambda value: value,
    )
    assert disabled._speech_rescue_enabled is False
