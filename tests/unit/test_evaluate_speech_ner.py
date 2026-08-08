from __future__ import annotations

from training.evaluate_ner import resolve_checkpoint
from training.evaluate_speech_ner import exact_metrics
from uzbek_speech_entities.ner.schemas import Entity


def test_exact_metrics_counts_entity_offsets_per_sample() -> None:
    gold = [
        (Entity(text="akmal", label="PER", start=0, end=5),),
        (Entity(text="toshkent", label="LOC", start=0, end=8),),
    ]
    predicted = [
        (Entity(text="akmal", label="PER", start=0, end=5),),
        (Entity(text="not", label="ORG", start=0, end=3),),
    ]

    metrics = exact_metrics(gold, predicted)

    assert metrics["overall"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "count": 2,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert metrics["PER"]["f1"] == 1.0
    assert metrics["LOC"]["recall"] == 0.0


def test_resolve_checkpoint_accepts_compact_promoted_directory(tmp_path) -> None:
    final = tmp_path / "final"
    final.mkdir()
    (final / "labels.json").write_text("{}", encoding="utf-8")

    checkpoint, artifact_root = resolve_checkpoint(final)

    assert checkpoint == final.resolve()
    assert artifact_root == final.resolve()
