from __future__ import annotations

import pytest
from pydantic import ValidationError

from uzbek_speech_entities.ner.schemas import Entity
from uzbek_speech_entities.pipeline.schemas import AnalysisModels, AnalysisResult, Timing


def test_entity_is_strict_and_bounded() -> None:
    entity = Entity(text="Akmal", label="PER", start=0, end=5, score=0.9)
    assert entity.label == "PER"
    with pytest.raises(ValidationError):
        Entity(text="", label="PER", start=0, end=0, score=1.1)
    with pytest.raises(ValidationError):
        Entity(text="Akmal", label="TEMPORAL", start=0, end=5, score=0.5)


def test_canonical_metadata_is_atomic_and_person_only() -> None:
    canonical = Entity(
        text="doskon",
        label="PER",
        start=0,
        end=6,
        canonical_text="Doston",
        canonical_confidence=0.83,
        canonical_source="name_lexicon_edit_distance",
    )
    assert canonical.text == "doskon"
    with pytest.raises(ValidationError, match="supplied together"):
        Entity(text="doskon", label="PER", start=0, end=6, canonical_text="Doston")
    with pytest.raises(ValidationError, match="only for PER"):
        Entity(
            text="doskon",
            label="ORG",
            start=0,
            end=6,
            canonical_text="Doston",
            canonical_confidence=0.83,
            canonical_source="name_lexicon_edit_distance",
        )


def test_analysis_result_is_uniform_for_text_or_audio() -> None:
    result = AnalysisResult(
        raw_transcript="Akmal",
        normalized_transcript="Akmal",
        entities=(Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),),
        timing=Timing(
            audio_preprocessing_ms=0,
            stt_ms=0,
            normalization_ms=1,
            ner_ms=1,
            total_ms=2,
        ),
        models=AnalysisModels(stt=None, ner="models/ner/final"),
    )
    assert result.models.stt is None
