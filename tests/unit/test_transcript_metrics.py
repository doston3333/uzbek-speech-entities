from __future__ import annotations

import pytest

from evaluation.dataset import GoldEntity
from evaluation.transcript_metrics import (
    aggregate_transcript_metrics,
    calculate_transcript_metrics,
    entity_mention_accuracy,
)


def test_raw_and_normalized_metrics_differ_and_corpus_is_weighted() -> None:
    sample = calculate_transcript_metrics("O‘quvchi, keldi!", "o'quvchi keldi")
    assert sample.raw_wer.rate > 0
    assert sample.normalized_wer.rate == 0
    assert sample.normalized_cer.rate == 0

    corpus = aggregate_transcript_metrics([("a", "x"), ("a b b b", "a b b b")])
    assert corpus.raw_wer.rate == pytest.approx(0.2)


def test_empty_and_invalid_transcript_inputs() -> None:
    empty = calculate_transcript_metrics("", "")
    assert empty.raw_wer.rate is None
    assert empty.normalized_cer.rate is None
    with pytest.raises(TypeError, match="reference"):
        calculate_transcript_metrics(None, "text")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        aggregate_transcript_metrics([])


def test_entity_mention_accuracy_is_multiset_capped_by_gold() -> None:
    entities = [
        GoldEntity("Akmal", "PER", 0, 5),
        GoldEntity("Akmal", "PER", 6, 11),
        GoldEntity("Toshkent", "LOC", 12, 20),
    ]
    scores = entity_mention_accuracy(entities, "Akmal Toshkent Toshkent Toshkent")
    assert scores["PER"].correct == 1
    assert scores["PER"].total == 2
    assert scores["LOC"].accuracy == 1.0
    assert scores["ORG"].accuracy is None
