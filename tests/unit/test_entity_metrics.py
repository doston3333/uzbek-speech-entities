from __future__ import annotations

import pytest

from evaluation.dataset import GoldEntity
from evaluation.entity_metrics import (
    align_entities_to_reference,
    calculate_entity_metrics,
    project_gold_entities,
)


def test_entity_metrics_are_multiset_and_macro_includes_all_four_labels() -> None:
    gold = [
        GoldEntity("Akmal", "PER", 0, 5),
        GoldEntity("Akmal", "PER", 6, 11),
        GoldEntity("Toshkent", "LOC", 12, 20),
    ]
    predicted = [
        GoldEntity("Akmal", "PER", 0, 5),
        GoldEntity("Akmal", "PER", 6, 11),
        GoldEntity("Toshkent", "ORG", 12, 20),
    ]
    metrics = calculate_entity_metrics(gold, predicted, mode="span")
    assert metrics.by_label["PER"].true_positives == 2
    assert metrics.by_label["LOC"].false_negatives == 1
    assert metrics.by_label["ORG"].false_positives == 1
    assert metrics.macro_f1 == pytest.approx(0.25)


def test_span_and_surface_modes_are_explicit() -> None:
    gold = [GoldEntity("Toshkent", "LOC", 0, 8)]
    prediction = [GoldEntity("toshkent", "LOC", 4, 12)]
    assert calculate_entity_metrics(gold, prediction, mode="span").overall.f1 == 0
    assert calculate_entity_metrics(gold, prediction, mode="surface").overall.f1 == 1
    with pytest.raises(ValueError, match="mode"):
        calculate_entity_metrics(gold, prediction, mode="invalid")  # type: ignore[arg-type]


def test_projection_handles_apostrophe_whitespace_and_date_normalization() -> None:
    text = "  O'quvchi   2026 - yil  5 - avgust keldi."
    entities = [
        GoldEntity("O'quvchi", "PER", 2, 10),
        GoldEntity("2026 - yil  5 - avgust", "DATE", 13, 35),
    ]
    projected = project_gold_entities(text, entities)
    assert [(item.text, item.start, item.end) for item in projected] == [
        ("Oʻquvchi", 0, 8),
        ("2026-yil 5-avgust", 9, 26),
    ]


def test_exact_token_alignment_projects_preserved_entity_tokens() -> None:
    predicted = [GoldEntity("Akmal Karimov", "PER", 0, 13)]

    projected = align_entities_to_reference(
        "Akmal Karimov Toshkentga bordi.",
        "Akmal Karimov toshkentga bordi",
        predicted,
    )

    assert [(item.text, item.label, item.start, item.end) for item in projected] == [
        ("Akmal Karimov", "PER", 0, 13)
    ]


def test_exact_token_alignment_uses_nonmatching_span_for_stt_substitution() -> None:
    predicted = [GoldEntity("Akmal Kerimov", "PER", 0, 13)]
    reference = "Akmal Karimov Toshkentga bordi."

    projected = align_entities_to_reference(
        reference,
        "Akmal Kerimov Toshkentga bordi",
        predicted,
    )

    assert projected[0].start > len(reference)
