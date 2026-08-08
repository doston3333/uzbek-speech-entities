"""Shared character and date mappings for Uzbek normalization."""

from typing import Final

from uzbek_speech_entities.constants import CANONICAL_APOSTROPHE

APOSTROPHE_VARIANTS: Final[tuple[str, ...]] = ("'", "’", "‘", "ʻ", "ʼ", "`")
_APOSTROPHE_REPLACEMENTS: Final[dict[str, int | str | None]] = {
    variant: CANONICAL_APOSTROPHE for variant in APOSTROPHE_VARIANTS
}
APOSTROPHE_TRANSLATION: Final[dict[int, int | str | None]] = str.maketrans(
    _APOSTROPHE_REPLACEMENTS
)

UZBEK_MONTHS: Final[tuple[str, ...]] = (
    "yanvar",
    "fevral",
    "mart",
    "aprel",
    "may",
    "iyun",
    "iyul",
    "avgust",
    "sentabr",
    "oktabr",
    "noyabr",
    "dekabr",
)

DATE_HYPHEN_SUFFIXES: Final[tuple[str, ...]] = ("yil", *UZBEK_MONTHS)
