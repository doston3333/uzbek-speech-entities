from __future__ import annotations

from pathlib import Path

from uzbek_speech_entities.audio.validation import AudioValidationConfig
from uzbek_speech_entities.ner.schemas import Entity
from uzbek_speech_entities.normalization import normalize_speech_analysis, project_analysis_span
from uzbek_speech_entities.pipeline.analyzer import SpeechEntityAnalyzer


def test_analysis_normalization_keeps_display_and_records_exact_source_spans() -> None:
    display = "oltinchi avgust kuni soati uchida ikki ming yigirma oltinchi yil"
    view = normalize_speech_analysis(display)

    assert view.display_text == display
    assert view.analysis_text == "6-avgust kuni soat 3 da 2026-yil"
    assert all(
        view.analysis_text[token.analysis_start : token.analysis_end] == token.text
        for token in view.tokens
    )
    assert all(
        token.source_start is None
        or display[token.source_start : token.source_end] for token in view.tokens
    )


def test_analysis_name_filler_rules_are_anchored_and_do_not_change_display() -> None:
    display = "mening ismim doskon am sardor bilan birga ko‘rishamiz doskon haqida"
    view = normalize_speech_analysis(display)

    assert display == view.display_text
    assert view.analysis_text == "mening ismim Doston, Sardor bilan birga ko‘rishamiz doskon haqida"
    assert "doskon haqida" in view.analysis_text
    filler = next(token for token in view.tokens if token.transformation == "filler_comma")
    assert filler.source_start is None and filler.source_end is None
    assert filler.hard_boundary_before and filler.hard_boundary_after


def test_exact_requested_analysis_view_and_strong_full_name_filler_context() -> None:
    display = (
        "assalomu alaykum mening ismim doskon am men bugun oltinchi avgust kuni "
        "alisher navoiy haqida gapiraman"
    )
    view = normalize_speech_analysis(display)

    assert view.display_text == display
    assert view.analysis_text == (
        "assalomu alaykum mening ismim Doston, men bugun 6-avgust kuni "
        "Alisher Navoiy haqida gapiraman"
    )
    assert normalize_speech_analysis("rajabov doston am men").analysis_text == (
        "Rajabov Doston, men"
    )
    assert normalize_speech_analysis("rajabov doston haqida").analysis_text == (
        "rajabov doston haqida"
    )


def test_clock_itn_preserves_whether_a_locative_was_spoken() -> None:
    assert normalize_speech_analysis("soat uch").analysis_text == "soat 3"
    assert normalize_speech_analysis("soati uchida").analysis_text == "soat 3 da"


def test_semantic_heads_truecase_specific_phrases_but_not_generic_heads() -> None:
    assert normalize_speech_analysis(
        "yangi oʻzbekiston universitetida"
    ).analysis_text == "Yangi Oʻzbekiston universitetida"
    assert normalize_speech_analysis("ashxobod parkida").analysis_text == "Ashxobod parkida"
    assert normalize_speech_analysis(
        "alisher navoiy nomidagi universitet"
    ).analysis_text == "Alisher Navoiy nomidagi universitet"
    assert normalize_speech_analysis("men universitetda oʻqiyman").analysis_text == (
        "men universitetda oʻqiyman"
    )
    assert normalize_speech_analysis("parkda sayr qildim").analysis_text == (
        "parkda sayr qildim"
    )


def test_projection_rejects_filler_crossing_and_collapsed_temporal_subspan() -> None:
    display = "oltinchi avgust am davom etadi"
    view = normalize_speech_analysis(display)
    temporal = view.tokens[0]
    comma = next(token for token in view.tokens if token.transformation == "filler_comma")

    partial_projection = project_analysis_span(
        view, temporal.analysis_start + 1, temporal.analysis_start + 2
    )
    assert partial_projection == (
        0,
        15,
    )
    assert project_analysis_span(view, temporal.analysis_start, comma.analysis_end) is None


def test_normalized_candidates_require_matching_label_and_project_display_offsets() -> None:
    display = "oltinchi avgust kuni"
    view = normalize_speech_analysis(display)
    entity = Entity(text="6", label="DATE", start=0, end=1, score=0.9)
    accepted = SpeechEntityAnalyzer._normalized_candidates(display, (entity,), view, 0.70)
    rejected = SpeechEntityAnalyzer._normalized_candidates(
        display,
        (entity.model_copy(update={"label": "PER"}),),
        view,
        0.70,
    )

    assert [(item.start, item.end, item.label, item.source) for item in accepted] == [
        (0, len(display), "DATE", "normalized_clean_model")
    ]
    assert rejected == ()


class _BatchOnlyPredictor:
    def __init__(self) -> None:
        self.inputs: list[tuple[str, ...]] = []

    @property
    def loaded(self) -> bool:
        return True

    @property
    def device(self) -> str | None:
        return "cpu"

    @property
    def model_path(self) -> Path:
        return Path("models/ner/fake")

    def load(self) -> None:
        return None

    def predict(self, _: str) -> tuple[Entity, ...]:
        raise AssertionError("analysis normalization must use predict_many")

    def predict_many(self, texts: tuple[str, ...]) -> tuple[tuple[Entity, ...], ...]:
        self.inputs.append(texts)
        return (), (Entity(text="6", label="DATE", start=0, end=1, score=0.9),)


def test_audio_analysis_uses_one_two_input_batch(monkeypatch, tmp_path: Path) -> None:
    from contextlib import contextmanager

    from fakes import FakeSTTService

    @contextmanager
    def fake_prepared_audio(_: Path, __: AudioValidationConfig):
        yield tmp_path / "canonical.wav"

    monkeypatch.setattr(
        "uzbek_speech_entities.pipeline.analyzer.prepared_audio", fake_prepared_audio
    )
    predictor = _BatchOnlyPredictor()
    analyzer = SpeechEntityAnalyzer(
        stt_service=FakeSTTService("oltinchi avgust"),
        ner_predictor=predictor,
        audio_config=AudioValidationConfig(16_000, True, 60, 1024, frozenset({"wav"})),
        normalizer=lambda text: text,
        speech_rescue_enabled=True,
        analysis_normalization_enabled=True,
    )

    result = analyzer.analyze_audio(tmp_path / "input.wav")
    assert predictor.inputs == [("oltinchi avgust", "6-avgust")]
    assert all(
        entity.text == result.normalized_transcript[entity.start : entity.end]
        for entity in result.entities
    )
