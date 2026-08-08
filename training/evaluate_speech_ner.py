"""Evaluate a candidate NER checkpoint on the immutable speech fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Any

if not __package__:
    # Direct script execution puts ``training/`` rather than the project root on
    # sys.path. Keep the documented CLI and package imports equally supported.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.evaluate_ner import resolve_checkpoint
from uzbek_speech_entities.config import load_config
from uzbek_speech_entities.ner.predictor import NERPredictor
from uzbek_speech_entities.ner.schemas import Entity, PublicEntityLabel
from uzbek_speech_entities.ner.span_resolver import resolve_candidates
from uzbek_speech_entities.ner.spans import validate_entity_spans
from uzbek_speech_entities.ner.speech_extractor import SpeechNERRescue
from uzbek_speech_entities.normalization import normalize_speech_analysis
from uzbek_speech_entities.pipeline.analyzer import SpeechEntityAnalyzer

DEFAULT_FIXTURE = Path("tests/fixtures/speech_ner_eval.jsonl")
LABELS: tuple[PublicEntityLabel, ...] = ("PER", "DATE", "ORG", "LOC")
NORMALIZED_CONFIDENCE_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Checkpoint, run, or pointer."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Override the application's display-model confidence threshold.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON metrics destination.")
    return parser.parse_args()


def _fixture(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("text"), str):
            raise ValueError(f"invalid fixture record at {path}:{number}")
        records.append(value)
    return records


def _gold(record: dict[str, Any]) -> tuple[Entity, ...]:
    values = record.get("entities")
    if not isinstance(values, list):
        raise ValueError(f"fixture record {record.get('id')!r} lacks entities")
    return validate_entity_spans(record["text"], tuple(Entity(**value) for value in values))


def _key(entity: Entity) -> tuple[str, int, int]:
    return entity.label, entity.start, entity.end


def exact_metrics(
    gold_samples: Sequence[Sequence[Entity]], predicted_samples: Sequence[Sequence[Entity]]
) -> dict[str, dict[str, float | int]]:
    """Calculate exact, per-sample entity metrics without cross-sample collisions."""
    if len(gold_samples) != len(predicted_samples):
        raise ValueError("gold and predicted sample counts differ")

    def counts(label: PublicEntityLabel | None) -> tuple[int, int, int]:
        true_positive = false_positive = false_negative = 0
        for gold, predicted in zip(gold_samples, predicted_samples, strict=True):
            gold_keys = Counter(_key(item) for item in gold if label is None or item.label == label)
            predicted_keys = Counter(
                _key(item) for item in predicted if label is None or item.label == label
            )
            shared = sum((gold_keys & predicted_keys).values())
            true_positive += shared
            false_positive += sum(predicted_keys.values()) - shared
            false_negative += sum(gold_keys.values()) - shared
        return true_positive, false_positive, false_negative

    def render(label: PublicEntityLabel | None) -> dict[str, float | int]:
        true_positive, false_positive, false_negative = counts(label)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "count": true_positive + false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {"overall": render(None), **{label: render(label) for label in LABELS}}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        with file.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _predictor(checkpoint: Path, confidence_threshold: float | None = None) -> NERPredictor:
    config = load_config()
    values = config.section("ner")
    return NERPredictor(
        checkpoint,
        max_length=int(values["max_length"]),
        confidence_threshold=(
            float(values["confidence_threshold"])
            if confidence_threshold is None
            else confidence_threshold
        ),
        visible_labels=tuple(values["visible_labels"]),
        model_to_application_labels=dict(values["model_to_application_labels"]),
        local_files_only=True,
    )


def evaluate_speech_checkpoint(
    checkpoint: Path,
    fixture_path: Path = DEFAULT_FIXTURE,
    *,
    predictor: NERPredictor | None = None,
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Run batched display/analysis views and return immutable-fixture metrics."""
    resolved, artifact_root = resolve_checkpoint(checkpoint)
    records = _fixture(fixture_path)
    fixture_before = fixture_path.read_bytes()
    views = [normalize_speech_analysis(record["text"]) for record in records]
    inputs = tuple(
        text
        for record, view in zip(records, views, strict=True)
        for text in (record["text"], view.analysis_text)
    )
    if predictor is not None and confidence_threshold is not None:
        raise ValueError("pass either predictor or confidence_threshold, not both")
    runner = predictor or _predictor(resolved, confidence_threshold)
    started = monotonic()
    predictions = runner.predict_many(inputs)
    if len(predictions) != len(inputs):
        raise RuntimeError("NER predictor returned an unexpected batch size")
    raw: list[tuple[Entity, ...]] = []
    projected: list[tuple[Entity, ...]] = []
    combined: list[tuple[Entity, ...]] = []
    rescue = SpeechNERRescue()
    transform_counts: Counter[str] = Counter()
    for index, (record, view) in enumerate(zip(records, views, strict=True)):
        display = validate_entity_spans(record["text"], predictions[index * 2])
        analysis = validate_entity_spans(view.analysis_text, predictions[index * 2 + 1])
        candidates = SpeechEntityAnalyzer._normalized_candidates(
            record["text"], analysis, view, NORMALIZED_CONFIDENCE_THRESHOLD
        )
        for token in view.tokens:
            if token.transformation != "identity":
                transform_counts[token.transformation] += 1
        raw.append(display)
        projected.append(resolve_candidates(record["text"], candidates))
        combined.append(
            validate_entity_spans(
                record["text"], rescue.extract(record["text"], display, candidates)
            )
        )
    if fixture_path.read_bytes() != fixture_before:
        raise RuntimeError("speech fixture changed during evaluation")
    gold = [_gold(record) for record in records]
    elapsed = monotonic() - started
    return {
        "checkpoint": {
            "resolved_path": str(resolved),
            "artifact_root": str(artifact_root),
            "sha256": _sha256(resolved),
        },
        "fixture": {
            "path": str(fixture_path),
            "sha256": hashlib.sha256(fixture_before).hexdigest(),
            "record_count": len(records),
        },
        "normalized_confidence_threshold": NORMALIZED_CONFIDENCE_THRESHOLD,
        "model_confidence_threshold": runner.confidence_threshold,
        "raw_display_model": exact_metrics(gold, raw),
        "normalized_only_projected_candidates": exact_metrics(gold, projected),
        "combined_final": exact_metrics(gold, combined),
        "transform_counts": dict(sorted(transform_counts.items())),
        "latency": {
            "total_seconds": elapsed,
            "mean_milliseconds": elapsed * 1000 / len(records) if records else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    metrics = evaluate_speech_checkpoint(
        args.checkpoint,
        args.fixture,
        confidence_threshold=args.confidence_threshold,
    )
    encoded = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
