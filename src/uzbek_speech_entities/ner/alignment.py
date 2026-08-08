"""Pure word-to-subword BIO alignment helpers for token classification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .labels import BIOValidationError, validate_bio_sequence

IGNORE_INDEX = -100


class LabelAlignmentError(ValueError):
    """Raised when a tokenizer word-ID sequence cannot align with BIO tags."""


@dataclass(frozen=True)
class TruncationStats:
    """Counts reported when a tokenizer drops a suffix of a token sequence."""

    examples: int = 0
    words: int = 0
    entity_words: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "truncated_entity_words": self.entity_words,
            "truncated_examples": self.examples,
            "truncated_words": self.words,
        }


def align_word_labels(
    word_ids: Sequence[int | None],
    ner_tags: Sequence[str],
    label2id: Mapping[str, int],
) -> list[int]:
    """Align word-level BIO labels to a tokenizer's word IDs.

    Special tokens and continuation subtokens use ``IGNORE_INDEX``.  The
    tokenizer may truncate only a suffix of words; malformed or non-contiguous
    word IDs are rejected rather than silently assigning an incorrect label.
    """
    tokens = [f"word-{index}" for index in range(len(ner_tags))]
    try:
        validate_bio_sequence(tokens, ner_tags)
    except BIOValidationError as error:
        raise LabelAlignmentError(str(error)) from error

    aligned: list[int] = []
    previous_word_id: int | None = None
    highest_word_id = -1
    seen_word_ids: set[int] = set()
    for token_index, word_id in enumerate(word_ids):
        if word_id is None:
            aligned.append(IGNORE_INDEX)
            continue
        if isinstance(word_id, bool) or not isinstance(word_id, int):
            raise LabelAlignmentError(f"invalid word ID at token index {token_index}: {word_id!r}")
        if word_id < 0 or word_id >= len(ner_tags):
            raise LabelAlignmentError(
                f"word ID {word_id} at token index {token_index} does not match "
                f"{len(ner_tags)} labels"
            )
        if previous_word_id is not None and word_id < previous_word_id:
            raise LabelAlignmentError("tokenizer word IDs must be nondecreasing")
        if word_id > highest_word_id + 1:
            raise LabelAlignmentError("tokenizer word IDs must not skip words")
        if word_id not in seen_word_ids:
            label = ner_tags[word_id]
            try:
                aligned.append(label2id[label])
            except KeyError as error:
                raise LabelAlignmentError(f"unknown label at word {word_id}: {label!r}") from error
            seen_word_ids.add(word_id)
            highest_word_id = word_id
        else:
            aligned.append(IGNORE_INDEX)
        previous_word_id = word_id
    return aligned


def truncation_stats(word_ids: Sequence[int | None], ner_tags: Sequence[str]) -> TruncationStats:
    """Return suffix-truncation counts after validating tokenizer word IDs."""
    # Alignment with a throwaway complete map provides the same strict word-ID
    # validation without coupling this accounting to a model's label IDs.
    full_map = {label: index for index, label in enumerate(dict.fromkeys(ner_tags))}
    align_word_labels(word_ids, ner_tags, full_map)
    retained = max((word_id for word_id in word_ids if word_id is not None), default=-1) + 1
    if retained >= len(ner_tags):
        return TruncationStats()
    dropped_tags = ner_tags[retained:]
    return TruncationStats(
        examples=1,
        words=len(dropped_tags),
        entity_words=sum(label != "O" for label in dropped_tags),
    )
