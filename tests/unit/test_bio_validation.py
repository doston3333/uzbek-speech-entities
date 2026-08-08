from __future__ import annotations

import pytest

from uzbek_speech_entities.ner.labels import (
    ENTITY_TYPES,
    VALID_BIO_LABELS,
    BIOValidationError,
    validate_bio_record,
    validate_bio_sequence,
)


def test_validator_accepts_every_supported_entity_type() -> None:
    tokens = [f"token-{entity_type}" for entity_type in ENTITY_TYPES]
    labels = [f"B-{entity_type}" for entity_type in ENTITY_TYPES]

    validate_bio_sequence(tokens, labels)
    assert {f"I-{entity_type}" for entity_type in ENTITY_TYPES}.issubset(VALID_BIO_LABELS)


@pytest.mark.parametrize(
    ("tokens", "labels", "message"),
    [
        (["Akmal"], ["I-PER"], "incompatible or orphan"),
        (["Akmal", "Toshkent"], ["B-PER", "I-LOC"], "incompatible or orphan"),
        (["Akmal"], ["B-UNKNOWN"], "unknown or malformed"),
        (["Akmal"], ["PER"], "unknown or malformed"),
        (["Akmal"], ["B_PER"], "unknown or malformed"),
        (["Akmal", "Karimov"], ["B-PER"], "length mismatch"),
        ([], [], "empty sentence"),
    ],
)
def test_validator_rejects_required_invalid_sequences(
    tokens: list[str], labels: list[str], message: str
) -> None:
    with pytest.raises(BIOValidationError, match=message):
        validate_bio_sequence(tokens, labels)


@pytest.mark.parametrize("record_id", [None, "", "   "])
def test_validator_rejects_missing_id(record_id: object) -> None:
    with pytest.raises(BIOValidationError, match="missing sentence ID"):
        validate_bio_record({"id": record_id, "tokens": ["Akmal"], "ner_tags": ["B-PER"]})
