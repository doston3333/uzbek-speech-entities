"""Additive canonicalization for person entities already accepted by rescue."""

from __future__ import annotations

from math import inf

from .offset_tokens import tokenize_words
from .rules.resources import load_person_names, load_person_phrases, load_resource
from .schemas import Entity


def _distance(left: str, right: str) -> int:
    """Return the Levenshtein edit distance for the short reviewed-name keys."""
    row = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        next_row = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            next_row.append(
                min(
                    next_row[-1] + 1,
                    row[right_index] + 1,
                    row[right_index - 1] + (left_character != right_character),
                )
            )
        row = next_row
    return row[-1]


def _maximum_distance(key: str) -> int:
    if len(key) <= 4:
        return 0
    if len(key) <= 7:
        return 1
    return 2


def reviewed_single_name(value: str) -> tuple[str, float] | None:
    """Return the uniquely-supported reviewed spelling for one name token.

    This is deliberately small and is shared by canonical metadata and the
    speech-only analysis view, so their accepted spelling corrections cannot
    drift apart.
    """
    key = tokenize_words(value)
    if len(key) != 1:
        return None
    comparison = key[0].comparison_key
    if comparison in _rejected_keys():
        return None
    maximum = _maximum_distance(comparison)
    scored = sorted(
        (_distance(comparison, name_key), canonical) for name_key, canonical in load_person_names()
    )
    best_distance, canonical = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else inf
    if best_distance > maximum or best_distance >= runner_up:
        return None
    confidence = 1.0 if best_distance == 0 else max(0.5, 1.0 - best_distance / len(comparison))
    return canonical, confidence


def _rejected_keys() -> frozenset[str]:
    heads = load_resource("heads_stopwords.json")
    temporal = load_resource("temporal_terms.json")
    return frozenset(
        {
            "assalomu",
            "alaykum",
            "salom",
            "xayr",
            "rahmat",
            *heads["person_reject"],
            *heads["organization_heads"],
            *heads["location_heads"],
            *temporal["months"],
            *temporal["weekdays"],
            *temporal["relative_dates"],
            *temporal["relative_modifiers"],
            *temporal["periods"],
        }
    )


def _canonical_for(entity: Entity) -> tuple[str, float, str] | None:
    if entity.label != "PER":
        return None
    keys = tuple(token.comparison_key for token in tokenize_words(entity.text))
    if not keys or any(key in _rejected_keys() for key in keys):
        return None
    for phrase_keys, canonical in load_person_phrases():
        if keys == phrase_keys:
            return canonical, 1.0, "person_phrase_gazetteer"
    if len(keys) != 1:
        return None
    reviewed = reviewed_single_name(entity.text)
    if reviewed is None:
        return None
    canonical, confidence = reviewed
    return canonical, confidence, "name_lexicon_edit_distance"


def canonicalize_person_entities(entities: tuple[Entity, ...]) -> tuple[Entity, ...]:
    """Attach canonical metadata without creating, moving, or relabeling entities."""
    canonicalized: list[Entity] = []
    for entity in entities:
        canonical = _canonical_for(entity)
        if canonical is None:
            canonicalized.append(entity)
            continue
        text, confidence, source = canonical
        canonicalized.append(
            entity.model_copy(
                update={
                    "canonical_text": text,
                    "canonical_confidence": confidence,
                    "canonical_source": source,
                }
            )
        )
    return tuple(canonicalized)
