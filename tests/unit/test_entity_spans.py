from __future__ import annotations

import pytest

from uzbek_speech_entities.ner.spans import (
    NERPredictionError,
    TokenPrediction,
    aggregate_bio_predictions,
    validate_entity_spans,
)


def _aggregate(text: str, predictions: list[TokenPrediction], threshold: float = 0.5):
    return aggregate_bio_predictions(
        text,
        predictions,
        model_to_application_labels={"PER": "PER", "LOC": "LOC", "ORG": "ORG", "TEMPORAL": "DATE"},
        visible_labels=frozenset({"PER", "LOC", "ORG", "DATE"}),
        threshold=threshold,
    )


@pytest.mark.parametrize(
    ("text", "predictions", "expected"),
    [
        ("Akmal", [TokenPrediction("B-PER", 0.9, 0, 5)], [("Akmal", "PER")]),
        (
            "Akmal Karimov",
            [TokenPrediction("B-PER", 0.9, 0, 5), TokenPrediction("I-PER", 0.8, 6, 13)],
            [("Akmal Karimov", "PER")],
        ),
        ("Toshkent", [TokenPrediction("B-LOC", 0.9, 0, 8)], [("Toshkent", "LOC")]),
        (
            "Toshkentga",
            [TokenPrediction("B-LOC", 0.9, 0, 10)],
            [("Toshkentga", "LOC")],
        ),
        (
            "Open Data Uzbekistan",
            [
                TokenPrediction("B-ORG", 0.9, 0, 4),
                TokenPrediction("I-ORG", 0.8, 5, 9),
                TokenPrediction("I-ORG", 0.7, 10, 20),
            ],
            [("Open Data Uzbekistan", "ORG")],
        ),
        (
            "Toshkent davlat universiteti",
            [
                TokenPrediction("B-ORG", 0.9, 0, 8),
                TokenPrediction("I-ORG", 0.9, 9, 15),
                TokenPrediction("I-ORG", 0.9, 16, 28),
            ],
            [("Toshkent davlat universiteti", "ORG")],
        ),
        (
            "2026-yil 5-avgust",
            [
                TokenPrediction("B-TEMPORAL", 0.9, 0, 8),
                TokenPrediction("I-TEMPORAL", 0.8, 9, 17),
            ],
            [("2026-yil 5-avgust", "DATE")],
        ),
        (
            "5-avgust kuni",
            [
                TokenPrediction("B-TEMPORAL", 0.9, 0, 8),
                TokenPrediction("I-TEMPORAL", 0.8, 9, 13),
            ],
            [("5-avgust kuni", "DATE")],
        ),
        (
            "kelasi hafta",
            [
                TokenPrediction("B-TEMPORAL", 0.9, 0, 6),
                TokenPrediction("I-TEMPORAL", 0.8, 7, 12),
            ],
            [("kelasi hafta", "DATE")],
        ),
        (
            "soat 15:30 da",
            [
                TokenPrediction("B-TEMPORAL", 0.9, 0, 4),
                TokenPrediction("I-TEMPORAL", 0.8, 5, 10),
                TokenPrediction("I-TEMPORAL", 0.7, 11, 13),
            ],
            [("soat 15:30 da", "DATE")],
        ),
    ],
)
def test_required_entity_span_shapes(
    text: str,
    predictions: list[TokenPrediction],
    expected: list[tuple[str, str]],
) -> None:
    entities = _aggregate(text, predictions)
    assert [(entity.text, entity.label) for entity in entities] == expected


def test_merges_malformed_bio_and_maps_temporal() -> None:
    text = "Akmal Karimov ertaga"
    entities = _aggregate(
        text,
        [
            TokenPrediction("I-PER", 0.8, 0, 5),
            TokenPrediction("I-PER", 0.6, 6, 13),
            TokenPrediction("B-TEMPORAL", 0.9, 14, 20),
        ],
    )
    assert [(entity.text, entity.label, entity.score) for entity in entities] == [
        ("Akmal Karimov", "PER", 0.7),
        ("ertaga", "DATE", 0.9),
    ]


def test_threshold_hidden_labels_and_overlap_are_deterministic() -> None:
    text = "Toshkent universiteti"
    entities = _aggregate(
        text,
        [
            TokenPrediction("B-LOC", 0.6, 0, 8),
            TokenPrediction("B-ORG", 0.9, 0, len(text)),
            TokenPrediction("B-MISC", 0.99, 9, len(text)),
        ],
        threshold=0.7,
    )
    assert [(entity.text, entity.label) for entity in entities] == [(text, "ORG")]


def test_unicode_apostrophe_offsets_and_span_validation() -> None:
    text = "Gʻijduvon Oʻzbekiston"
    entities = _aggregate(text, [TokenPrediction("B-LOC", 0.9, 0, 9)])
    assert entities[0].text == "Gʻijduvon"
    with pytest.raises(NERPredictionError):
        validate_entity_spans(text, [entities[0].model_copy(update={"start": 1})])


def test_invalid_token_offsets_are_rejected() -> None:
    with pytest.raises(NERPredictionError):
        TokenPrediction("B-PER", 0.9, -1, 2)
    with pytest.raises(NERPredictionError):
        _aggregate("Ali", [TokenPrediction("B-PER", 0.9, 0, 4)])
