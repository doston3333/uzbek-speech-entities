from __future__ import annotations

from uzbek_speech_entities.ner.spans import TokenPrediction, aggregate_bio_predictions


def test_only_public_mapped_labels_are_returned() -> None:
    entities = aggregate_bio_predictions(
        "Ali Toshkent Acme bugun pul son misc",
        [
            TokenPrediction("B-PER", 0.9, 0, 3),
            TokenPrediction("B-LOC", 0.9, 4, 12),
            TokenPrediction("B-ORG", 0.9, 13, 17),
            TokenPrediction("B-TEMPORAL", 0.9, 18, 23),
            TokenPrediction("B-MONEY", 0.99, 24, 27),
            TokenPrediction("B-NUMERIC", 0.99, 28, 31),
            TokenPrediction("B-MISC", 0.99, 32, 36),
        ],
        model_to_application_labels={
            "PER": "PER",
            "LOC": "LOC",
            "ORG": "ORG",
            "TEMPORAL": "DATE",
        },
        visible_labels=frozenset({"PER", "LOC", "ORG", "DATE"}),
        threshold=0.5,
    )
    assert [(entity.text, entity.label) for entity in entities] == [
        ("Ali", "PER"),
        ("Toshkent", "LOC"),
        ("Acme", "ORG"),
        ("bugun", "DATE"),
    ]
