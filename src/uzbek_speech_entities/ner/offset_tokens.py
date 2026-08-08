"""Offset-preserving lexical tokens for rule-based speech recovery."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+(?:[ʻ’‘ʼ'`:/.-]\w+)*", re.UNICODE)
_APOSTROPHES = str.maketrans("", "", "ʻ’‘ʼ'`ˈʹ´＇")


def comparison_key(value: str) -> str:
    """Return an Uzbek case- and apostrophe-insensitive comparison key."""
    return unicodedata.normalize("NFKC", value).casefold().translate(_APOSTROPHES)


@dataclass(frozen=True, slots=True)
class WordToken:
    """An immutable word and its half-open offsets in the displayed transcript."""

    text: str
    start: int
    end: int
    boundary_before: bool = False
    comparison_key: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.text or self.start < 0 or self.end <= self.start:
            raise ValueError("WordToken requires a non-empty, valid offset span")
        object.__setattr__(self, "comparison_key", comparison_key(self.text))


def tokenize_words(text: str) -> tuple[WordToken, ...]:
    """Tokenize words without changing the text whose offsets are returned."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens: list[WordToken] = []
    previous_end = 0
    for match in _WORD.finditer(text):
        gap = text[previous_end : match.start()]
        tokens.append(
            WordToken(
                match.group(),
                match.start(),
                match.end(),
                boundary_before=any(not character.isspace() for character in gap),
            )
        )
        previous_end = match.end()
    return tuple(tokens)
