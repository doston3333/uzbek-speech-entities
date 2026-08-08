from __future__ import annotations

import numpy as np

from uzbek_speech_entities.ner.labels import ENTITY_TYPES, build_label_maps
from uzbek_speech_entities.ner.training_metrics import compute_ner_metrics


def test_metrics_filter_ignore_index_and_report_entity_counts_and_macro_f1() -> None:
    label2id, id2label = build_label_maps()
    labels = np.array(
        [
            [-100, label2id["B-PER"], label2id["I-PER"], -100],
            [-100, label2id["B-LOC"], label2id["O"], -100],
        ]
    )
    predicted_ids = np.array(
        [
            [label2id["O"], label2id["B-PER"], label2id["I-PER"], label2id["O"]],
            [label2id["B-ORG"], label2id["O"], label2id["O"], label2id["B-ORG"]],
        ]
    )
    logits = np.full((2, 4, len(label2id)), -10.0)
    for batch_index, row in enumerate(predicted_ids):
        for token_index, predicted_id in enumerate(row):
            logits[batch_index, token_index, predicted_id] = 10.0

    metrics = compute_ner_metrics(logits, labels, id2label)

    assert metrics["PER_count"] == 1
    assert metrics["LOC_count"] == 1
    assert metrics["ORG_count"] == 0
    assert metrics["TEMPORAL_count"] == 0
    assert metrics["PER_f1"] == 1.0
    assert metrics["LOC_f1"] == 0.0
    assert metrics["four_class_macro_f1"] == 0.25
    assert metrics["token_accuracy"] == 0.75
    assert metrics["overall_f1"] == 2 / 3
    for entity_type in ("PER", "LOC", "ORG", "TEMPORAL", "MISC", "MONEY", "NUMERIC", "WORK"):
        assert {f"{entity_type}_{name}" for name in ("precision", "recall", "f1", "count")} <= set(
            metrics
        )


def test_metrics_return_zeroes_when_every_position_is_ignored() -> None:
    label2id, id2label = build_label_maps()

    metrics = compute_ner_metrics(
        np.zeros((1, 2, len(label2id))),
        np.array([[-100, -100]]),
        id2label,
    )

    assert metrics["overall_precision"] == 0.0
    assert metrics["overall_recall"] == 0.0
    assert metrics["overall_f1"] == 0.0
    assert metrics["token_accuracy"] == 0.0
    assert metrics["four_class_macro_f1"] == 0.0
    assert all(metrics[f"{entity_type}_count"] == 0 for entity_type in ENTITY_TYPES)
