from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from training.evaluate_public_speech_years import (
    evaluate_public_speech_checkpoint,
    score_public_speech_years,
)
from uzbek_speech_entities.ner.schemas import Entity, PublicEntityLabel


def _entity(
    text: str, start: int, end: int, label: PublicEntityLabel = "DATE"
) -> Entity:
    return Entity(text=text[start:end], label=label, start=start, end=end, score=0.9)


def test_public_speech_year_scoring_distinguishes_full_and_partial_date_spans() -> None:
    texts = ["ikki ming va besh yilni", "bir ming toʻqqiz yuz yili"]
    predictions = [
        (_entity(texts[0], 0, len(texts[0])),),
        (_entity(texts[1], 0, len("bir ming")),),
    ]
    metrics = score_public_speech_years(texts, predictions)
    assert metrics == {
        "record_count": 2,
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "partial_date_record_count": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_public_speech_year_evaluation_uses_phrase_only_fixture_and_fake_predictor(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "labels.json").write_text("{}\n", encoding="utf-8")
    fixture = tmp_path / "dev.jsonl"
    fixture.write_text(
        json.dumps(
            {
                "id": "year",
                "tokens": ["ikki", "ming", "va", "besh", "yilni"],
                "ner_tags": [
                    "B-TEMPORAL",
                    "I-TEMPORAL",
                    "I-TEMPORAL",
                    "I-TEMPORAL",
                    "I-TEMPORAL",
                ],
                "source": "fixture",
                "augmentation": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class Predictor:
        confidence_threshold = 0.8
        loaded = True
        device = "cpu"
        model_path = checkpoint

        def load(self) -> None:
            return None

        def predict(self, text: str) -> tuple[Entity, ...]:
            return self.predict_many((text,))[0]

        def predict_many(self, texts: Sequence[str]) -> tuple[tuple[Entity, ...], ...]:
            return tuple((_entity(text, 0, len(text)),) for text in texts)

    report = evaluate_public_speech_checkpoint(checkpoint, fixture, predictor=Predictor())
    assert report["raw_date_exact_span"]["recall"] == 1.0
    assert report["raw_date_exact_span"]["partial_date_record_count"] == 0


def test_public_speech_year_fixture_rejects_non_temporal_labels(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "labels.json").write_text("{}\n", encoding="utf-8")
    fixture = tmp_path / "bad.jsonl"
    fixture.write_text(
        '{"id":"bad","tokens":["ikki"],"ner_tags":["O"]}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="phrase-only TEMPORAL"):
        evaluate_public_speech_checkpoint(checkpoint, fixture, predictor=None)


def test_public_speech_year_fixture_must_not_be_empty(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "labels.json").write_text("{}\n", encoding="utf-8")
    fixture = tmp_path / "empty.jsonl"
    fixture.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_public_speech_checkpoint(checkpoint, fixture, predictor=None)
