"""Deterministic validation and conflict resolution for rescue candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .offset_tokens import tokenize_words
from .schemas import Entity, EntitySource, PublicEntityLabel

_PRIORITY: Final[dict[str, int]] = {
    "temporal_grammar": 0,
    "gazetteer_boundary_expansion": 1,
    "person_gazetteer": 2,
    "person_introduction": 3,
    "model_boundary_expansion": 4,
    "clean_model": 5,
    "normalized_clean_model": 6,
    "person_relation": 7,
}
_GREETINGS: Final[frozenset[str]] = frozenset({"assalomu", "alaykum", "salom", "xayr", "rahmat"})
_MAX_WORDS: Final[dict[PublicEntityLabel, int]] = {"PER": 3, "LOC": 6, "ORG": 6, "DATE": 12}


@dataclass(frozen=True, slots=True)
class Candidate:
    """An internal candidate that can be checked before becoming public output."""

    label: PublicEntityLabel
    start: int
    end: int
    source: EntitySource
    score: float | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or self.source not in _PRIORITY:
            raise ValueError("invalid speech NER candidate")
        if self.score is not None and not 0.0 <= self.score <= 1.0:
            raise ValueError("candidate score must be in [0, 1]")


def _valid(text: str, candidate: Candidate) -> bool:
    if candidate.end > len(text):
        return False
    all_tokens = tokenize_words(text)
    if not any(token.start == candidate.start for token in all_tokens) or not any(
        token.end == candidate.end for token in all_tokens
    ):
        return False
    covered = tuple(
        token
        for token in all_tokens
        if candidate.start <= token.start and token.end <= candidate.end
    )
    maximum_words = (
        4
        if candidate.label == "PER" and candidate.source == "person_gazetteer"
        else _MAX_WORDS[candidate.label]
    )
    if not covered or len(covered) > maximum_words:
        return False
    if candidate.label == "PER" and any(token.comparison_key in _GREETINGS for token in covered):
        return False
    return True


def resolve_candidates(
    text: str, candidates: tuple[Candidate, ...] | list[Candidate]
) -> tuple[Entity, ...]:
    """Keep exact, capped, non-overlapping candidates according to fixed priorities."""
    valid = [candidate for candidate in candidates if _valid(text, candidate)]
    deduplicated: dict[tuple[PublicEntityLabel, int, int], tuple[int, Candidate]] = {}
    for position, candidate in enumerate(valid):
        key = (candidate.label, candidate.start, candidate.end)
        current = deduplicated.get(key)
        if current is None:
            deduplicated[key] = (position, candidate)
            continue
        current_position, current_candidate = current
        candidate_score = candidate.score if candidate.score is not None else -1.0
        current_score = current_candidate.score if current_candidate.score is not None else -1.0
        if (_PRIORITY[candidate.source], -candidate_score, position) < (
            _PRIORITY[current_candidate.source],
            -current_score,
            current_position,
        ):
            deduplicated[key] = (position, candidate)
    selected: list[Candidate] = []
    for candidate in sorted(
        (candidate for _, candidate in deduplicated.values()),
        key=lambda item: (
            _PRIORITY[item.source],
            item.start,
            -item.end,
            item.label,
            item.source,
        ),
    ):
        if any(candidate.start < other.end and other.start < candidate.end for other in selected):
            continue
        selected.append(candidate)
    return tuple(
        Entity(
            text=text[candidate.start : candidate.end],
            label=candidate.label,
            start=candidate.start,
            end=candidate.end,
            score=candidate.score,
            source=candidate.source,
            evidence=candidate.evidence or None,
        )
        for candidate in sorted(selected, key=lambda item: (item.start, item.end, item.label))
    )
