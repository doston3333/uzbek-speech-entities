"""Conservative, offset-aligned analysis normalization for speech transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..ner.canonicalize_person import reviewed_single_name
from ..ner.offset_tokens import comparison_key, tokenize_words
from ..ner.rules.boundaries import semantic_head_label
from ..ner.rules.resources import load_person_phrases, load_resource
from ..ner.rules.temporal import anchored_year_value, cardinal_number, ordinal_number
from .aligned_tokens import AlignedToken, AnalysisNormalization, TransformationType

_LEXICAL_UNITS = re.compile(r"\w+(?:[ʻ’‘ʼ'`:/.-]\w+)*|[^\w\s]", re.UNICODE)
_NO_SPACE_BEFORE = frozenset(",.;:!?%…)]}»")
_FILLERS = frozenset({"am", "um", "mmm", "ee", "eee"})
_REPLACEMENT_PRIORITY = {
    "person_phrase": 0,
    "person_name": 1,
    "organization_case": 2,
    "location_case": 2,
    "temporal_itn": 3,
    "filler_comma": 4,
}


@dataclass(frozen=True, slots=True)
class _SourceUnit:
    text: str
    start: int
    end: int
    key: str


@dataclass(frozen=True, slots=True)
class _Replacement:
    start: int
    end: int
    text: str
    transformation: TransformationType
    confidence: float
    source_start: int | None
    source_end: int | None


def _units(text: str) -> tuple[_SourceUnit, ...]:
    return tuple(
        _SourceUnit(match.group(), match.start(), match.end(), comparison_key(match.group()))
        for match in _LEXICAL_UNITS.finditer(text)
    )


def _is_word(unit: _SourceUnit) -> bool:
    return bool(unit.text) and (unit.text[0].isalnum() or unit.text[0] == "_")


def _surname_like(key: str) -> bool:
    return len(key) >= 5 and key.endswith(
        ("ov", "ev", "yev", "ova", "eva", "yeva", "zoda", "ogli", "qizi")
    )


def _temporal_replacements(units: tuple[_SourceUnit, ...]) -> tuple[_Replacement, ...]:
    replacements: list[_Replacement] = []
    months = frozenset(load_resource("temporal_terms.json")["months"])
    index = 0
    while index < len(units):
        unit = units[index]
        ordinal = ordinal_number(unit.key)
        if ordinal is not None and 1 <= ordinal <= 31 and index + 1 < len(units):
            month = units[index + 1]
            if month.key in months and _is_word(month):
                replacements.append(
                    _Replacement(
                        index,
                        index + 2,
                        f"{ordinal}-{month.key}",
                        "temporal_itn",
                        1.0,
                        unit.start,
                        month.end,
                    )
                )
                index += 2
                continue
        if unit.key in {"soat", "soati", "soatda"} and index + 1 < len(units):
            value_key = units[index + 1].key
            value = cardinal_number(value_key)
            locative = unit.key == "soatda"
            if value is None:
                for suffix in ("ida", "da", "ta"):
                    if value_key.endswith(suffix):
                        value = cardinal_number(value_key[: -len(suffix)])
                        locative = value is not None
                        break
            if value is not None and 1 <= value <= 24:
                rendered = f"soat {value} da" if locative else f"soat {value}"
                replacements.append(
                    _Replacement(
                        index,
                        index + 2,
                        rendered,
                        "temporal_itn",
                        1.0,
                        unit.start,
                        units[index + 1].end,
                    )
                )
                index += 2
                continue
        if unit.key == "yil":
            for width in range(min(7, index), 1, -1):
                start = index - width
                year = anchored_year_value(tuple(item.key for item in units[start:index]))
                if year is not None:
                    replacements.append(
                        _Replacement(
                            start,
                            index + 1,
                            f"{year}-yil",
                            "temporal_itn",
                            1.0,
                            units[start].start,
                            unit.end,
                        )
                    )
                    break
        index += 1
    return tuple(replacements)


def _person_replacements(units: tuple[_SourceUnit, ...]) -> tuple[_Replacement, ...]:
    replacements: list[_Replacement] = []
    for index in range(len(units)):
        for phrase, canonical in load_person_phrases():
            end = index + len(phrase)
            if end <= len(units) and tuple(unit.key for unit in units[index:end]) == phrase:
                replacements.append(
                    _Replacement(
                        index,
                        end,
                        canonical,
                        "person_phrase",
                        1.0,
                        units[index].start,
                        units[end - 1].end,
                    )
                )

    # A surname-shaped token followed by an exact reviewed name and a filler is
    # a narrow, speech-specific full-name context.  Keep the two source spans
    # separate so a model prediction for only the given name cannot project to
    # the whole pair.
    for index in range(len(units) - 2):
        surname, given_name, following = units[index : index + 3]
        if not _surname_like(surname.key) or following.key not in _FILLERS:
            continue
        reviewed = reviewed_single_name(given_name.text)
        if reviewed is None or comparison_key(given_name.text) != comparison_key(reviewed[0]):
            continue
        replacements.extend(
            (
                _Replacement(
                    index,
                    index + 1,
                    surname.text[:1].upper() + surname.text[1:],
                    "person_name",
                    0.95,
                    surname.start,
                    surname.end,
                ),
                _Replacement(
                    index + 1,
                    index + 2,
                    reviewed[0],
                    "person_name",
                    reviewed[1],
                    given_name.start,
                    given_name.end,
                ),
            )
        )
    social_verbs = frozenset(load_resource("heads_stopwords.json")["social_verbs"])
    for index, unit in enumerate(units):
        if not _is_word(unit):
            continue
        intro = index > 0 and units[index - 1].key == "ismim" and (
            index < 2 or units[index - 2].key in {"men", "mening"}
        )
        relation = (
            index + 1 < len(units)
            and units[index + 1].key == "bilan"
            and any(item.key in social_verbs for item in units[index + 2 : index + 8])
        )
        if not intro and not relation:
            continue
        reviewed = reviewed_single_name(unit.text)
        if reviewed is None:
            continue
        canonical, confidence = reviewed
        if relation and comparison_key(unit.text) != comparison_key(canonical):
            continue
        replacements.append(
            _Replacement(
                index,
                index + 1,
                canonical,
                "person_name",
                confidence,
                unit.start,
                unit.end,
            )
        )
    return tuple(replacements)


def _semantic_head_replacements(
    display_text: str, units: tuple[_SourceUnit, ...]
) -> tuple[_Replacement, ...]:
    """Truecase only a short phrase whose final token is an ORG/LOC head."""
    words = tokenize_words(display_text)
    if not words:
        return ()
    resource = load_resource("heads_stopwords.json")
    safe_modifiers = frozenset(resource["safe_modifiers"])
    blocked = frozenset(resource["boundary_stopwords"]) | {"nomidagi"}
    temporal = load_resource("temporal_terms.json")
    blocked |= frozenset(
        item
        for field in ("months", "weekdays", "relative_dates", "relative_modifiers", "periods")
        for item in temporal[field]
    )
    unit_indexes = {(unit.start, unit.end): index for index, unit in enumerate(units)}
    replacements: list[_Replacement] = []
    for head_index, head in enumerate(words):
        label = semantic_head_label(head)
        if label is None or head_index == 0 or head.boundary_before:
            continue
        prior = words[head_index - 1]
        if prior.comparison_key in blocked or not prior.comparison_key.isalpha():
            continue
        start_index = head_index - 1
        if head_index >= 2:
            earlier = words[head_index - 2]
            include_earlier = (
                earlier.comparison_key in safe_modifiers
                or (
                    label == "LOC"
                    and reviewed_single_name(prior.text) is not None
                    and earlier.comparison_key not in blocked
                    and earlier.comparison_key.isalpha()
                )
            )
            if include_earlier and not prior.boundary_before:
                start_index = head_index - 2
        selected_words = words[start_index : head_index + 1]
        source_unit_start = unit_indexes.get(
            (selected_words[0].start, selected_words[0].end)
        )
        source_unit_end = unit_indexes.get((head.start, head.end))
        if source_unit_start is None or source_unit_end is None:
            continue
        rendered_words = [
            word.text[:1].upper() + word.text[1:] if index < len(selected_words) - 1 else word.text
            for index, word in enumerate(selected_words)
        ]
        replacements.append(
            _Replacement(
                source_unit_start,
                source_unit_end + 1,
                " ".join(rendered_words),
                "organization_case" if label == "ORG" else "location_case",
                0.95,
                selected_words[0].start,
                head.end,
            )
        )
    return tuple(replacements)


def _filler_replacements(units: tuple[_SourceUnit, ...]) -> tuple[_Replacement, ...]:
    return tuple(
        _Replacement(index, index + 1, ",", "filler_comma", 1.0, None, None)
        for index, unit in enumerate(units)
        if _is_word(unit) and unit.key in _FILLERS
    )


def _select_replacements(
    display_text: str, units: tuple[_SourceUnit, ...]
) -> dict[int, _Replacement]:
    selected: dict[int, _Replacement] = {}
    occupied: set[int] = set()
    ordered = (
        *_person_replacements(units),
        *_semantic_head_replacements(display_text, units),
        *_temporal_replacements(units),
        *_filler_replacements(units),
    )
    for replacement in sorted(
        ordered,
        key=lambda item: (
            _REPLACEMENT_PRIORITY[item.transformation],
            item.start,
            -(item.end - item.start),
        ),
    ):
        if any(index in occupied for index in range(replacement.start, replacement.end)):
            continue
        selected[replacement.start] = replacement
        occupied.update(range(replacement.start, replacement.end))
    return selected


def _render_piece(
    rendered: list[AlignedToken],
    output: list[str],
    text: str,
    source_start: int | None,
    source_end: int | None,
    transformation: TransformationType,
    confidence: float,
    *,
    hard_boundary: bool = False,
) -> None:
    if output and text not in _NO_SPACE_BEFORE:
        output.append(" ")
    start = sum(len(part) for part in output)
    output.append(text)
    rendered.append(
        AlignedToken(
            text=text,
            analysis_start=start,
            analysis_end=start + len(text),
            source_start=source_start,
            source_end=source_end,
            transformation=transformation,
            confidence=confidence,
            hard_boundary_before=hard_boundary,
            hard_boundary_after=hard_boundary,
        )
    )


def normalize_speech_analysis(display_text: str) -> AnalysisNormalization:
    """Build a conservative model-facing view without changing display text."""
    if not isinstance(display_text, str):
        raise TypeError("display_text must be a string")
    units = _units(display_text)
    selected = _select_replacements(display_text, units)
    rendered: list[AlignedToken] = []
    output: list[str] = []
    index = 0
    while index < len(units):
        replacement = selected.get(index)
        if replacement is None:
            unit = units[index]
            _render_piece(
                rendered,
                output,
                unit.text,
                unit.start,
                unit.end,
                "identity",
                1.0,
            )
            index += 1
            continue
        _render_piece(
            rendered,
            output,
            replacement.text,
            replacement.source_start,
            replacement.source_end,
            replacement.transformation,
            replacement.confidence,
            hard_boundary=replacement.transformation == "filler_comma",
        )
        index = replacement.end
    return AnalysisNormalization(display_text, "".join(output), tuple(rendered))
