from __future__ import annotations

from pathlib import Path

import pytest

from uzbek_speech_entities.ner.predictor import NERPredictor
from uzbek_speech_entities.stt.base import ModelLoadError


def test_lexical_units_preserve_full_words_punctuation_and_unicode_offsets() -> None:
    units = NERPredictor.lexical_units("Akmal, Gʻijduvonga 2026-yil!")
    assert units == (
        ("Akmal", 0, 5),
        (",", 5, 6),
        ("Gʻijduvonga", 7, 18),
        ("2026-yil", 19, 27),
        ("!", 27, 28),
    )


def test_model_load_failure_is_typed_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = NERPredictor(
        Path("missing"),
        max_length=128,
        confidence_threshold=0.5,
        visible_labels=("PER", "LOC", "ORG", "DATE"),
        model_to_application_labels={"PER": "PER"},
        local_files_only=True,
    )
    calls = 0

    def fail_dependencies() -> tuple[object, object, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("private model location")

    monkeypatch.setattr(predictor, "_dependencies", fail_dependencies)
    with pytest.raises(ModelLoadError) as first_error:
        predictor.load()
    assert "private" not in str(first_error.value)
    with pytest.raises(ModelLoadError):
        predictor.load()
    assert calls == 1


@pytest.mark.slow
def test_local_final_model_public_ambiguity_examples() -> None:
    model_path = Path("models/ner/final")
    if not model_path.exists():
        pytest.skip("local final NER artifact is not available")
    from uzbek_speech_entities.config import load_config

    predictor = NERPredictor.from_config(load_config(), local_files_only=True)
    examples = {
        "Akmal Karimov Toshkentga bordi.": [
            ("Akmal Karimov", "PER"),
            ("Toshkentga", "LOC"),
        ],
        "Toshkent shahriga bordi.": [("Toshkent shahriga", "LOC")],
        "Toshkent davlat texnika universitetiga bordi.": [
            ("Toshkent davlat texnika universitetiga", "ORG")
        ],
        "U 100 ta kitob oldi.": [],
        "U 2026-yil 5-avgust kuni keldi.": [("2026-yil 5-avgust kuni", "DATE")],
    }
    for text, expected in examples.items():
        entities = predictor.predict(text)
        assert [(entity.text, entity.label) for entity in entities] == expected
        assert all(entity.label in {"PER", "LOC", "ORG", "DATE"} for entity in entities)

    long_text = " ".join(["oddiy"] * 180)
    assert len(NERPredictor.lexical_units(long_text)) > predictor.max_length
    assert isinstance(predictor.predict(long_text), tuple)
