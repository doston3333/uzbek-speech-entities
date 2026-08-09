"""Exact multi-token person phrases for precision-first speech recovery."""

from __future__ import annotations

from dataclasses import dataclass

from ..offset_tokens import WordToken
from ..span_resolver import Candidate
from .boundaries import semantic_head_label
from .resources import load_person_phrases


@dataclass(frozen=True, slots=True)
class PersonPhraseMatch:
    """One exact phrase match expressed as token indexes."""

    start: int
    end: int
    canonical: str


def person_phrase_matches(tokens: tuple[WordToken, ...]) -> tuple[PersonPhraseMatch, ...]:
    """Match only reviewed two-to-four-token phrases against comparison keys."""
    matches: list[PersonPhraseMatch] = []
    for index in range(len(tokens)):
        for keys, canonical in load_person_phrases():
            end = index + len(keys)
            phrase_tokens = tokens[index:end]
            if (
                end <= len(tokens)
                and all(not token.boundary_before for token in phrase_tokens[1:])
                and tuple(token.comparison_key for token in phrase_tokens) == keys
            ):
                matches.append(PersonPhraseMatch(index, end, canonical))
    return tuple(matches)


def person_gazetteer_candidates(tokens: tuple[WordToken, ...]) -> tuple[Candidate, ...]:
    """Emit exact phrase people, retaining the original transcript offsets."""
    return tuple(
        Candidate(
            label="PER",
            start=tokens[match.start].start,
            end=tokens[match.end - 1].end,
            source="person_gazetteer",
            evidence=("exact_person_phrase",),
        )
        for match in person_phrase_matches(tokens)
    )


def gazetteer_boundary_expansion_candidates(tokens: tuple[WordToken, ...]) -> tuple[Candidate, ...]:
    """Let a phrase plus an immediate semantic head win as one ORG/LOC span."""
    candidates: list[Candidate] = []
    for match in person_phrase_matches(tokens):
        head_index = match.end
        if head_index < len(tokens) and tokens[head_index].comparison_key == "nomidagi":
            head_index += 1
        if head_index >= len(tokens) or tokens[head_index].boundary_before:
            continue
        label = semantic_head_label(tokens[head_index])
        if label is None:
            continue
        candidates.append(
            Candidate(
                label=label,
                start=tokens[match.start].start,
                end=tokens[head_index].end,
                source="gazetteer_boundary_expansion",
                evidence=("person_phrase_semantic_head",),
            )
        )
    return tuple(candidates)
