"""Hard-anchored Uzbek DATE grammar for noisy speech transcripts."""

from __future__ import annotations

import re

from ..offset_tokens import WordToken
from ..span_resolver import Candidate
from .resources import load_resource

_NUMERIC_DATE = re.compile(r"\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?$")
_NUMERIC_TIME = re.compile(r"\d{1,2}:\d{2}$")
_TEMPORAL_SUFFIXES = ("gacha", "idan", "ida", "dan", "da", "de", "ta", "te", "ga")
_YEAR_ANCHORS = frozenset(
    {"yil", "yili", "yilda", "yilida", "yilning", "yildan", "yilgacha", "yilni"}
)
_YEAR_NUMBERS = {
    "bir": 1,
    "ikki": 2,
    "uch": 3,
    "tort": 4,
    "besh": 5,
    "olti": 6,
    "yetti": 7,
    "sakkiz": 8,
    "toqqiz": 9,
    "on": 10,
    "yigirma": 20,
    "ottiz": 30,
    "qirq": 40,
    "ellik": 50,
    "oltmish": 60,
    "yetmish": 70,
    "sakson": 80,
    "toqson": 90,
    "yuz": 100,
    "ming": 1000,
}
_YEAR_ORDINALS = {
    "birinchi": 1,
    "ikkinchi": 2,
    "uchinchi": 3,
    "tortinchi": 4,
    "beshinchi": 5,
    "oltinchi": 6,
    "yettinchi": 7,
    "sakkizinchi": 8,
    "toqqizinchi": 9,
    "oninchi": 10,
    "yigirmanchi": 20,
    "ottizinchi": 30,
    "qirqinchi": 40,
    "elliginchi": 50,
    "oltmishinchi": 60,
    "yetmishinchi": 70,
    "saksoninchi": 80,
    "toqsoninchi": 90,
    "yuzinchi": 100,
    "minginchi": 1000,
}
_YEAR_TOKENS = {**_YEAR_NUMBERS, **_YEAR_ORDINALS}


def _terms() -> dict[str, frozenset[str]]:
    return {
        name: frozenset(values) for name, values in load_resource("temporal_terms.json").items()
    }


def _is_ordinal(key: str, terms: dict[str, frozenset[str]]) -> bool:
    return key in terms["ordinal_words"]


def _with_allowed_suffix(key: str, values: frozenset[str]) -> bool:
    return key in values or any(
        key.removesuffix(suffix) in values for suffix in _TEMPORAL_SUFFIXES if key.endswith(suffix)
    )


def _is_month(key: str, terms: dict[str, frozenset[str]]) -> bool:
    return _with_allowed_suffix(key, terms["months"])


def _is_time_value(token: WordToken, terms: dict[str, frozenset[str]]) -> bool:
    key = token.comparison_key
    return (
        key.isdecimal()
        or bool(_NUMERIC_TIME.fullmatch(key))
        or _with_allowed_suffix(key, terms["number_words"])
        or key in terms["time_specials"]
    )


def _time_value_count(
    tokens: tuple[WordToken, ...], start: int, terms: dict[str, frozenset[str]]
) -> int:
    count = 0
    while start + count < len(tokens) and count < 4:
        key = tokens[start + count].comparison_key
        if not _is_time_value(tokens[start + count], terms):
            break
        count += 1
        if (
            bool(_NUMERIC_TIME.fullmatch(key))
            or key in terms["time_specials"]
            or any(key.endswith(suffix) for suffix in _TEMPORAL_SUFFIXES)
        ):
            break
    return count


def _unknown_bridge(token: WordToken, terms: dict[str, frozenset[str]]) -> bool:
    key = token.comparison_key
    forbidden = terms["bridge_stopwords"] | terms["number_words"] | terms["relative_modifiers"]
    return key.isalpha() and 1 <= len(key) <= 4 and key not in forbidden


def _is_numeric_calendar(key: str, terms: dict[str, frozenset[str]]) -> bool:
    if _NUMERIC_DATE.fullmatch(key):
        return True
    if re.fullmatch(r"\d{4}-yil(?:da|dan|ga|gacha)?", key):
        return True
    match = re.fullmatch(r"(?P<day>\d{1,2})-(?P<month>[a-z]+)", key)
    if match is None or not 1 <= int(match.group("day")) <= 31:
        return False
    return _is_month(match.group("month"), terms)


def _number_prefix_start(
    tokens: tuple[WordToken, ...], month_index: int, terms: dict[str, frozenset[str]]
) -> int:
    """Include at most three number words ending in an ordinal before a month."""
    if month_index == 0 or not _is_ordinal(tokens[month_index - 1].comparison_key, terms):
        return month_index
    start = month_index - 1
    while (
        start > 0
        and month_index - start < 3
        and tokens[start - 1].comparison_key in terms["number_words"]
    ):
        start -= 1
    return start


def _calendar_end(
    tokens: tuple[WordToken, ...], index: int, terms: dict[str, frozenset[str]]
) -> int:
    """Consume only tails and a complete clock phrase after a calendar anchor."""
    end = index + 1
    while end < len(tokens):
        key = tokens[end].comparison_key
        if key in terms["date_tails"]:
            end += 1
            continue
        if key in terms["time_markers"]:
            count = _time_value_count(tokens, end + 1, terms)
            if count:
                end += 1 + count
            break
        break
    return end


def _edit_distance(left: str, right: str) -> int:
    """Return a small Levenshtein distance for one anchored number token."""
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


def _fuzzy_year_token(key: str) -> str | None:
    # Common ASR contraction of ``yigirma``; keep this correction anchored to a year.
    if key == "yirma":
        return "yigirma"
    matches = [known for known in _YEAR_TOKENS if _edit_distance(key, known) == 1]
    return matches[0] if len(matches) == 1 else None


def _year_value(keys: tuple[str, ...]) -> int | None:
    """Parse a contiguous Uzbek number/ordinal phrase into a plausible year."""
    total = 0
    group = 0
    for position, key in enumerate(keys):
        value = _YEAR_TOKENS[key]
        if key in _YEAR_ORDINALS and position != len(keys) - 1:
            return None
        if value == 1000:
            if not 1 <= group <= 2:
                return None
            total += group * value
            group = 0
        elif value == 100:
            if not 1 <= group <= 9:
                return None
            group *= value
        else:
            group += value
    year = total + group
    return year if 1000 <= year <= 2199 else None


def ordinal_number(key: str) -> int | None:
    """Return an exact reviewed Uzbek ordinal value, if it is one."""
    return _YEAR_ORDINALS.get(key)


def cardinal_number(key: str) -> int | None:
    """Return an exact reviewed Uzbek cardinal value, if it is one."""
    return _YEAR_NUMBERS.get(key)


def anchored_year_value(keys: tuple[str, ...]) -> int | None:
    """Parse an exact Uzbek year phrase without ASR fuzzy correction."""
    if not keys or any(key not in _YEAR_TOKENS for key in keys):
        return None
    return _year_value(keys)


def _normalized_year_keys(tokens: tuple[WordToken, ...]) -> tuple[str, ...] | None:
    keys: list[str] = []
    corrections = 0
    for position, token in enumerate(tokens):
        if position > 0 and token.boundary_before:
            return None
        key = token.comparison_key
        if key in _YEAR_TOKENS:
            keys.append(key)
            continue
        corrected = _fuzzy_year_token(key) if corrections == 0 else None
        if corrected is None:
            return None
        corrections += 1
        keys.append(corrected)
    return tuple(keys)


def _anchored_years(tokens: tuple[WordToken, ...]) -> dict[int, tuple[int, tuple[str, ...]]]:
    """Find valid two-to-seven-token year phrases ending at a hard year anchor."""
    years: dict[int, tuple[int, tuple[str, ...]]] = {}
    for anchor, token in enumerate(tokens):
        if token.comparison_key not in _YEAR_ANCHORS:
            continue
        maximum_width = min(7, anchor)
        for width in range(maximum_width, 1, -1):
            start = anchor - width
            keys = _normalized_year_keys(tokens[start:anchor])
            if keys is not None and _year_value(keys) is not None:
                years[anchor] = (start, ("anchored_year",))
                break
    return years


def temporal_candidates(tokens: tuple[WordToken, ...]) -> tuple[Candidate, ...]:
    """Find DATE spans only where a concrete Uzbek temporal grammar anchors them."""
    terms = _terms()
    anchored_years = _anchored_years(tokens)
    anchored_year_starts = {
        start: (anchor, evidence) for anchor, (start, evidence) in anchored_years.items()
    }
    candidates: list[Candidate] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        key = token.comparison_key
        start = index
        end: int | None = None
        evidence: tuple[str, ...] = ()

        if index in anchored_years:
            year_start, evidence = anchored_years[index]
            start, end = year_start, index + 1
        elif _is_ordinal(key, terms) and index + 1 < len(tokens):
            if _is_month(tokens[index + 1].comparison_key, terms):
                start = _number_prefix_start(tokens, index + 1, terms)
                end, evidence = _calendar_end(tokens, index + 1, terms), ("ordinal_month",)
            elif (
                index + 2 < len(tokens)
                and _unknown_bridge(tokens[index + 1], terms)
                and _is_month(tokens[index + 2].comparison_key, terms)
            ):
                end, evidence = _calendar_end(tokens, index + 2, terms), ("ordinal_bridge_month",)
        elif key in terms["relative_modifiers"] and index + 1 < len(tokens):
            following = tokens[index + 1].comparison_key
            if following in terms["weekdays"] or following in terms["periods"]:
                end, evidence = _calendar_end(tokens, index + 1, terms), ("relative_modifier",)
        elif _is_month(key, terms):
            start = _number_prefix_start(tokens, index, terms)
            end, evidence = _calendar_end(tokens, index, terms), ("calendar_anchor",)
        elif key in terms["weekdays"] or key in terms["relative_dates"]:
            end, evidence = _calendar_end(tokens, index, terms), ("calendar_anchor",)
        elif _is_numeric_calendar(key, terms):
            calendar_index = index
            if index + 1 < len(tokens) and _is_numeric_calendar(
                tokens[index + 1].comparison_key, terms
            ):
                calendar_index = index + 1
            end, evidence = _calendar_end(tokens, calendar_index, terms), ("numeric_date",)
        elif key in terms["time_markers"]:
            count = _time_value_count(tokens, index + 1, terms)
            if count:
                end, evidence = index + 1 + count, ("clock_marker",)
        elif _NUMERIC_TIME.fullmatch(key):
            end, evidence = index + 1, ("numeric_time",)

        if end is None:
            index += 1
            continue
        if end in anchored_year_starts and start < end:
            anchor, year_evidence = anchored_year_starts[end]
            end = anchor + 1
            evidence = (*evidence, *year_evidence)
        candidates.append(
            Candidate(
                label="DATE",
                start=tokens[start].start,
                end=tokens[end - 1].end,
                source="temporal_grammar",
                evidence=evidence,
            )
        )
        index = end
    return tuple(candidates)
