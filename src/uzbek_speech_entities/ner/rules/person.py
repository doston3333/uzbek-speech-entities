"""Conservative Uzbek person-introduction and relation rules."""

from __future__ import annotations

from ..offset_tokens import WordToken
from ..span_resolver import Candidate
from .resources import load_name_lexicon, load_resource


def _sets() -> dict[str, frozenset[str]]:
    resource = load_resource("heads_stopwords.json")
    temporal = load_resource("temporal_terms.json")
    temporal_boundaries = frozenset(
        item
        for field in (
            "months",
            "weekdays",
            "relative_dates",
            "relative_modifiers",
            "periods",
            "date_tails",
            "time_markers",
        )
        for item in temporal[field]
    )
    return {
        **{name: frozenset(values) for name, values in resource.items()},
        "intro_boundaries": frozenset(resource["intro_boundaries"]) | temporal_boundaries,
        "names": load_name_lexicon(),
    }


def _is_name_word(token: WordToken) -> bool:
    return token.comparison_key.isalpha()


def person_introduction_candidates(tokens: tuple[WordToken, ...]) -> tuple[Candidate, ...]:
    """Recover a short name after ``mening ismim`` or ``ismim`` only."""
    terms = _sets()
    candidates: list[Candidate] = []
    for index, token in enumerate(tokens):
        if token.comparison_key != "ismim":
            continue
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor].comparison_key in terms["intro_fillers"]:
            cursor += 1
        collected: list[WordToken] = []
        while cursor < len(tokens) and len(collected) < 3:
            current = tokens[cursor]
            key = current.comparison_key
            if (
                current.boundary_before
                or key in terms["intro_fillers"]
                or key in terms["intro_boundaries"]
                or key in terms["person_reject"]
                or not _is_name_word(current)
            ):
                break
            collected.append(current)
            cursor += 1
        if not collected:
            continue
        if (
            len(collected) == 3
            and cursor < len(tokens)
            and not tokens[cursor].boundary_before
            and _is_name_word(tokens[cursor])
            and tokens[cursor].comparison_key not in terms["intro_boundaries"]
        ):
            continue
        if cursor < len(tokens) and tokens[cursor].comparison_key == "emas":
            continue
        candidates.append(
            Candidate(
                label="PER",
                start=collected[0].start,
                end=collected[-1].end,
                source="person_introduction",
                evidence=("ismim_introduction",),
            )
        )
    return tuple(candidates)


def _surname_like(key: str) -> bool:
    return len(key) >= 5 and key.endswith(
        ("ov", "ev", "yev", "ova", "eva", "yeva", "zoda", "ogli", "qizi")
    )


def person_relation_candidates(tokens: tuple[WordToken, ...]) -> tuple[Candidate, ...]:
    """Recover an evidenced person immediately before a social ``bilan`` phrase."""
    terms = _sets()
    candidates: list[Candidate] = []
    for index, token in enumerate(tokens):
        if token.comparison_key != "bilan" or token.boundary_before or index == 0:
            continue
        person = tokens[index - 1]
        key = person.comparison_key
        if key not in terms["names"] and not _surname_like(key):
            continue
        next_index = index + 1
        if next_index < len(tokens) and tokens[next_index].comparison_key == "birga":
            next_index += 1
        lookahead = tokens[next_index : next_index + 6]
        if not any(word.comparison_key in terms["social_verbs"] for word in lookahead):
            continue
        candidates.append(
            Candidate(
                label="PER",
                start=person.start,
                end=person.end,
                source="person_relation",
                evidence=("bilan_social_verb",),
            )
        )
    return tuple(candidates)
