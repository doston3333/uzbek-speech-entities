"""Normalization used exclusively for transcript error metrics."""

from __future__ import annotations

import unicodedata

from uzbek_speech_entities.constants import CANONICAL_APOSTROPHE

from .runtime import normalize_runtime


def _replace_punctuation(text: str) -> str:
    characters: list[str] = []
    last_index = len(text) - 1

    for index, character in enumerate(text):
        if character == CANONICAL_APOSTROPHE:
            is_inside_word = (
                index > 0
                and index < last_index
                and text[index - 1].isalnum()
                and text[index + 1].isalnum()
            )
            characters.append(character if is_inside_word else " ")
        elif unicodedata.category(character).startswith("P"):
            characters.append(" ")
        else:
            characters.append(character)

    return "".join(characters)


def normalize_evaluation(text: str) -> str:
    """Return lowercase, punctuation-free text for WER and CER calculation.

    This representation must never be used for UI display or NER offsets.
    Uzbek apostrophes inside words are retained in their canonical form while
    quotation marks and other punctuation become word boundaries.
    """
    normalized = normalize_runtime(text).lower()
    normalized = _replace_punctuation(normalized)
    return " ".join(normalized.split())


normalize_for_evaluation = normalize_evaluation
