"""Entity-first NER metrics with lazy NumPy and seqeval imports."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .labels import ENTITY_TYPES


def compute_ner_metrics(
    predictions: Any,
    label_ids: Any,
    id2label: Mapping[int, str],
) -> dict[str, float | int]:
    """Compute seqeval entity metrics and masked token accuracy.

    ``predictions`` may be logits or already-selected token IDs.  Positions
    with the standard ``-100`` ignore label are excluded from every metric.
    """
    import numpy as np
    from seqeval.metrics import (  # type: ignore[import-untyped]
        accuracy_score,
        classification_report,
    )

    predicted_ids = np.asarray(predictions)
    expected_ids = np.asarray(label_ids)
    if predicted_ids.ndim == expected_ids.ndim + 1:
        predicted_ids = predicted_ids.argmax(axis=-1)
    if predicted_ids.shape != expected_ids.shape:
        raise ValueError(
            f"prediction and label shapes differ: {predicted_ids.shape!r} != {expected_ids.shape!r}"
        )

    true_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    for prediction_row, label_row in zip(predicted_ids, expected_ids, strict=True):
        true_row: list[str] = []
        predicted_row: list[str] = []
        for predicted_id, label_id in zip(prediction_row, label_row, strict=True):
            if int(label_id) == -100:
                continue
            try:
                true_row.append(id2label[int(label_id)])
                predicted_row.append(id2label[int(predicted_id)])
            except KeyError as error:
                raise ValueError(f"unknown prediction or label ID: {error.args[0]!r}") from error
        if true_row:
            true_sequences.append(true_row)
            predicted_sequences.append(predicted_row)

    report = (
        classification_report(
            true_sequences,
            predicted_sequences,
            output_dict=True,
            zero_division=0,
        )
        if true_sequences
        else {}
    )
    metrics: dict[str, float | int] = {
        "overall_precision": float(report.get("micro avg", {}).get("precision", 0.0)),
        "overall_recall": float(report.get("micro avg", {}).get("recall", 0.0)),
        "overall_f1": float(report.get("micro avg", {}).get("f1-score", 0.0)),
        "token_accuracy": float(accuracy_score(true_sequences, predicted_sequences))
        if true_sequences
        else 0.0,
    }
    for entity_type in ENTITY_TYPES:
        entity_metrics = report.get(entity_type, {})
        metrics[f"{entity_type}_precision"] = float(entity_metrics.get("precision", 0.0))
        metrics[f"{entity_type}_recall"] = float(entity_metrics.get("recall", 0.0))
        metrics[f"{entity_type}_f1"] = float(entity_metrics.get("f1-score", 0.0))
        metrics[f"{entity_type}_count"] = int(entity_metrics.get("support", 0))
    metrics["four_class_macro_f1"] = sum(
        float(metrics[f"{entity_type}_f1"])
        for entity_type in ("PER", "LOC", "ORG", "TEMPORAL")
    ) / 4
    return metrics


def trainer_compute_metrics(
    id2label: Mapping[int, str],
) -> Callable[[Any], dict[str, float | int]]:
    """Create the Trainer callback while keeping Trainer imports out of this module."""

    def compute(eval_prediction: Any) -> dict[str, float | int]:
        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return compute_ner_metrics(predictions, eval_prediction.label_ids, id2label)

    return compute
