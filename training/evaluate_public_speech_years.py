"""Evaluate raw model DATE spans on held-out public spoken-year phrases."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.evaluate_ner import resolve_checkpoint
from uzbek_speech_entities.config import load_config
from uzbek_speech_entities.ner.labels import validate_bio_record
from uzbek_speech_entities.ner.predictor import NERPredictor, NERService
from uzbek_speech_entities.ner.schemas import Entity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.80)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank held-out speech-year line at {path}:{line_number}")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"non-object held-out speech-year record at {path}:{line_number}")
        validate_bio_record(record)
        tags = record["ner_tags"]
        if not tags or tags[0] != "B-TEMPORAL" or any(tag != "I-TEMPORAL" for tag in tags[1:]):
            raise ValueError(
                f"held-out speech-year record must be phrase-only TEMPORAL: {record.get('id')}"
            )
        records.append(record)
    if not records:
        raise ValueError("held-out public speech-year fixture must not be empty")
    return records


def _predictor(checkpoint: Path, confidence_threshold: float) -> NERPredictor:
    values = load_config().section("ner")
    return NERPredictor(
        checkpoint,
        max_length=int(values["max_length"]),
        confidence_threshold=confidence_threshold,
        visible_labels=tuple(values["visible_labels"]),
        model_to_application_labels=dict(values["model_to_application_labels"]),
        local_files_only=True,
    )


def score_public_speech_years(
    texts: Sequence[str], predictions: Sequence[Sequence[Entity]]
) -> dict[str, float | int]:
    if len(texts) != len(predictions):
        raise ValueError("held-out texts and predictions have different lengths")
    if not texts:
        raise ValueError("held-out public speech-year fixture must not be empty")
    true_positive = false_positive = false_negative = partial_records = 0
    for text, entities in zip(texts, predictions, strict=True):
        predicted_dates = [entity for entity in entities if entity.label == "DATE"]
        exact = sum(entity.start == 0 and entity.end == len(text) for entity in predicted_dates)
        shared = min(1, exact)
        true_positive += shared
        false_positive += len(predicted_dates) - shared
        false_negative += 1 - shared
        if not shared and predicted_dates:
            partial_records += 1
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = true_positive / len(texts) if texts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "record_count": len(texts),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "partial_date_record_count": partial_records,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_public_speech_checkpoint(
    checkpoint: Path,
    fixture_path: Path,
    *,
    confidence_threshold: float = 0.80,
    predictor: NERService | None = None,
) -> dict[str, Any]:
    """Measure full-phrase raw DATE extraction without the rescue layer."""
    resolved, _artifact_root = resolve_checkpoint(checkpoint)
    records = _records(fixture_path)
    texts = [" ".join(record["tokens"]) for record in records]
    runner = predictor or _predictor(resolved, confidence_threshold)
    predictions = runner.predict_many(texts)
    return {
        "checkpoint": str(resolved),
        "confidence_threshold": confidence_threshold,
        "fixture_path": str(fixture_path),
        "fixture_sha256": _sha256(fixture_path),
        "raw_date_exact_span": score_public_speech_years(texts, predictions),
    }


def main() -> None:
    args = parse_args()
    report = evaluate_public_speech_checkpoint(
        args.checkpoint,
        args.fixture,
        confidence_threshold=args.confidence_threshold,
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
