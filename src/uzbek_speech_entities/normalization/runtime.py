"""Conservative normalization for UI display and NER input."""

from __future__ import annotations

import re
import unicodedata

from .mappings import APOSTROPHE_TRANSLATION, UZBEK_MONTHS

_WHITESPACE_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(r"\s+([,.;:!?%…\)\]\}»])")
_TIME_COLON_RE = re.compile(r"(?<=\d)\s*:\s*(?=\d)")
_MONTH_PATTERN = "|".join(re.escape(month) for month in UZBEK_MONTHS)
_YEAR_HYPHEN_RE = re.compile(
    r"\b(?P<number>\d{4})\s*-\s*(?P<suffix>yil)\b",
    flags=re.IGNORECASE,
)
_DAY_MONTH_HYPHEN_RE = re.compile(
    rf"\b(?P<number>\d{{1,2}})\s*-\s*(?P<suffix>{_MONTH_PATTERN})\b",
    flags=re.IGNORECASE,
)


def _normalize_day_month_hyphen(match: re.Match[str]) -> str:
    day = int(match.group("number"))
    if not 1 <= day <= 31:
        return match.group(0)
    return f"{match.group('number')}-{match.group('suffix')}"


def normalize_runtime(text: str) -> str:
    """Normalize surface formatting without changing words, case, or meaning.

    The result is safe to display and to pass to NER. The transformation is
    deliberately conservative and idempotent; it does not correct spelling or
    resolve relative temporal expressions.
    """
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.translate(APOSTROPHE_TRANSLATION)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    normalized = _YEAR_HYPHEN_RE.sub(r"\g<number>-\g<suffix>", normalized)
    normalized = _DAY_MONTH_HYPHEN_RE.sub(_normalize_day_month_hyphen, normalized)
    normalized = _TIME_COLON_RE.sub(":", normalized)
    normalized = _SPACE_BEFORE_PUNCTUATION_RE.sub(r"\1", normalized)
    return normalized.strip()
