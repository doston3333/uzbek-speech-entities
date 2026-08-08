"""Strict BIO label validation shared by NER dataset tooling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

ENTITY_TYPES: Final[tuple[str, ...]] = (
    "LOC",
    "ORG",
    "PER",
    "MISC",
    "TEMPORAL",
    "NUMERIC",
    "WORK",
    "MONEY",
)
_ENTITY_TYPE_SET: Final[frozenset[str]] = frozenset(ENTITY_TYPES)
VALID_BIO_LABELS: Final[frozenset[str]] = frozenset(
    ("O", *(f"{prefix}-{entity_type}" for prefix in ("B", "I") for entity_type in ENTITY_TYPES))
)

# Keep the outside label first, then make the rest reproducible regardless of
# source-file ordering.  The prepared gold corpus contains this full 17-label
# vocabulary; training must never infer a smaller map from an individual split.
CANONICAL_BIO_LABELS: Final[tuple[str, ...]] = (
    "O",
    *(f"B-{entity_type}" for entity_type in ENTITY_TYPES),
    *(f"I-{entity_type}" for entity_type in ENTITY_TYPES),
)


class BIOValidationError(ValueError):
    """Raised when a record does not satisfy the strict prepared BIO schema."""


def build_label_maps(
    observed_labels: Sequence[str] | None = None,
) -> tuple[dict[str, int], dict[int, str]]:
    """Return the deterministic full BIO vocabulary used by all NER stages.

    ``observed_labels`` is accepted to make accidental label loss explicit: all
    supplied labels must be part of the supported vocabulary.  We intentionally
    keep labels absent from a particular split so model heads and metrics are
    compatible across train, validation, and test.
    """
    if observed_labels is not None:
        unknown = sorted({label for label in observed_labels if label not in VALID_BIO_LABELS})
        if unknown:
            raise BIOValidationError(f"unknown labels for mapping: {unknown!r}")
    label2id = {label: index for index, label in enumerate(CANONICAL_BIO_LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def validate_bio_sequence(tokens: Sequence[str], ner_tags: Sequence[str]) -> None:
    """Validate a non-empty token sequence and its strict BIO tags.

    ``I-X`` must immediately continue an active entity of the same type.  This
    deliberately rejects, rather than repairs, malformed source annotations.
    """
    if isinstance(tokens, str | bytes) or isinstance(ner_tags, str | bytes):
        raise BIOValidationError("tokens and ner_tags must be sequences, not strings")
    if not tokens:
        raise BIOValidationError("empty sentence")
    if len(tokens) != len(ner_tags):
        raise BIOValidationError("token and label length mismatch")

    active_entity_type: str | None = None
    for index, (token, label) in enumerate(zip(tokens, ner_tags, strict=True)):
        if not isinstance(token, str) or not token:
            raise BIOValidationError(f"invalid token at index {index}")
        if not isinstance(label, str) or label not in VALID_BIO_LABELS:
            raise BIOValidationError(f"unknown or malformed label at index {index}: {label!r}")
        if label == "O":
            active_entity_type = None
            continue

        prefix, entity_type = label.split("-", maxsplit=1)
        if entity_type not in _ENTITY_TYPE_SET:
            raise BIOValidationError(f"unknown entity type at index {index}: {entity_type!r}")
        if prefix == "I" and active_entity_type != entity_type:
            raise BIOValidationError(
                f"incompatible or orphan I-{entity_type} at index {index} "
                f"after {active_entity_type!r}"
            )
        active_entity_type = entity_type


def validate_bio_record(record: Mapping[str, object]) -> None:
    """Validate the required prepared-record fields, including a nonblank ID."""
    record_id = record.get("id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise BIOValidationError("missing sentence ID")

    tokens = record.get("tokens")
    ner_tags = record.get("ner_tags")
    if not isinstance(tokens, Sequence) or isinstance(tokens, str | bytes):
        raise BIOValidationError("tokens must be a sequence")
    if not isinstance(ner_tags, Sequence) or isinstance(ner_tags, str | bytes):
        raise BIOValidationError("ner_tags must be a sequence")
    validate_bio_sequence(tokens, ner_tags)
