from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Any

import pytest

from uzbek_speech_entities.ner.schemas import Entity, PublicEntityLabel
from uzbek_speech_entities.ner.spans import validate_entity_spans
from uzbek_speech_entities.ner.speech_extractor import SpeechNERRescue
from uzbek_speech_entities.normalization import normalize_speech_analysis
from uzbek_speech_entities.pipeline.analyzer import SpeechEntityAnalyzer

FIXTURE = Path(__file__).parents[1] / "fixtures" / "speech_ner_eval.jsonl"
LABELS: tuple[PublicEntityLabel, ...] = ("PER", "LOC", "ORG", "DATE")


def _records() -> list[dict[str, Any]]:
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]


def _entity(value: dict[str, Any], *, model: bool) -> Entity:
    return Entity(
        text=value["text"],
        label=value["label"],
        start=value["start"],
        end=value["end"],
        score=value.get("score") if model else None,
    )


def _key(entity: Entity) -> tuple[str, int, int]:
    return (entity.label, entity.start, entity.end)


def _score(
    gold_by_sample: list[tuple[Entity, ...]],
    predicted_by_sample: list[tuple[Entity, ...]],
    label: PublicEntityLabel | None = None,
) -> tuple[float, float, float]:
    true_positives = false_positives = false_negatives = 0
    for gold, predicted in zip(gold_by_sample, predicted_by_sample, strict=True):
        gold_keys = Counter(_key(item) for item in gold if label is None or item.label == label)
        predicted_keys = Counter(
            _key(item) for item in predicted if label is None or item.label == label
        )
        true_positives += sum((gold_keys & predicted_keys).values())
        false_positives += sum(predicted_keys.values()) - sum(
            (gold_keys & predicted_keys).values()
        )
        false_negatives += sum(gold_keys.values()) - sum((gold_keys & predicted_keys).values())
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def test_speech_fixture_meets_size_balance_and_hard_negative_requirements() -> None:
    records = _records()
    assert 40 <= len(records) <= 60
    assert sum(bool(record["hard_negative"]) for record in records) >= 20
    counts = Counter(entity["label"] for record in records for entity in record["entities"])
    assert counts == {"PER": 15, "DATE": 15, "ORG": 10, "LOC": 10}


def test_speech_rescue_exact_span_metrics_and_latency_meet_local_gate() -> None:
    records = _records()
    rescue = SpeechNERRescue()
    gold_by_sample: list[tuple[Entity, ...]] = []
    baseline_by_sample: list[tuple[Entity, ...]] = []
    rescued_by_sample: list[tuple[Entity, ...]] = []

    for record in records:
        text = record["text"]
        gold = tuple(_entity(value, model=False) for value in record["entities"])
        baseline = tuple(_entity(value, model=True) for value in record["model_entities"])
        rescued = rescue.extract(text, baseline)
        validate_entity_spans(text, gold)
        validate_entity_spans(text, baseline)
        validate_entity_spans(text, rescued)
        assert [_key(entity) for entity in rescued] == [
            _key(entity) for entity in gold
        ], record["id"]
        gold_by_sample.append(gold)
        baseline_by_sample.append(baseline)
        rescued_by_sample.append(rescued)

    overall_precision, _, _ = _score(gold_by_sample, rescued_by_sample)
    date_precision, date_recall, _ = _score(gold_by_sample, rescued_by_sample, "DATE")
    _, baseline_date_recall, _ = _score(gold_by_sample, baseline_by_sample, "DATE")
    _, per_recall, _ = _score(gold_by_sample, rescued_by_sample, "PER")
    _, baseline_per_recall, _ = _score(gold_by_sample, baseline_by_sample, "PER")
    rescued_macro_f1 = sum(
        _score(gold_by_sample, rescued_by_sample, label)[2] for label in LABELS
    ) / len(LABELS)
    baseline_macro_f1 = sum(
        _score(gold_by_sample, baseline_by_sample, label)[2] for label in LABELS
    ) / len(LABELS)

    assert overall_precision >= 0.90
    assert date_precision >= 0.95
    assert rescued_macro_f1 - baseline_macro_f1 >= 0.10
    assert per_recall - baseline_per_recall >= 0.20
    assert date_recall - baseline_date_recall >= 0.20

    started = monotonic()
    repetitions = 20
    for _ in range(repetitions):
        for record, baseline in zip(records, baseline_by_sample, strict=True):
            rescue.extract(record["text"], baseline)
    mean_milliseconds = (monotonic() - started) * 1000 / (len(records) * repetitions)
    assert mean_milliseconds < 10.0


@pytest.mark.slow
def test_local_model_normalized_only_candidates_meet_precision_gate() -> None:
    """Keep transformed-view evidence precision-first on the speech fixture."""
    model_path = Path("models/ner/final")
    if not model_path.exists():
        pytest.skip("local final NER artifact is not available")
    from uzbek_speech_entities.config import load_config
    from uzbek_speech_entities.ner.predictor import NERPredictor

    records = _records()
    views = [normalize_speech_analysis(record["text"]) for record in records]
    model_inputs = tuple(
        text
        for record, view in zip(records, views, strict=True)
        for text in (record["text"], view.analysis_text)
    )
    predictions = NERPredictor.from_config(load_config(), local_files_only=True).predict_many(
        model_inputs
    )
    true_positives = false_positives = false_negatives = 0
    for index, (record, view) in enumerate(zip(records, views, strict=True)):
        normalized_predictions = predictions[index * 2 + 1]
        projected = SpeechEntityAnalyzer._normalized_candidates(
            record["text"], normalized_predictions, view, 0.70
        )
        predicted_keys = {(item.label, item.start, item.end) for item in projected}
        gold_keys = {
            (item["label"], item["start"], item["end"]) for item in record["entities"]
        }
        true_positives += len(predicted_keys & gold_keys)
        false_positives += len(predicted_keys - gold_keys)
        false_negatives += len(gold_keys - predicted_keys)
        assert all(
            record["text"][item.start : item.end]
            and item.start < item.end <= len(record["text"])
            for item in projected
        )

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    assert precision >= 0.95
    assert recall >= 0.20
