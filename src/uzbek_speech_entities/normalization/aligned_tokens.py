"""Immutable metadata for a model-facing speech analysis transcript."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransformationType = Literal[
    "identity",
    "person_phrase",
    "person_name",
    "organization_case",
    "location_case",
    "temporal_itn",
    "filler_comma",
]


@dataclass(frozen=True, slots=True)
class AlignedToken:
    """One rendered lexical unit and the display span from which it came."""

    text: str
    analysis_start: int
    analysis_end: int
    source_start: int | None
    source_end: int | None
    transformation: TransformationType
    confidence: float
    hard_boundary_before: bool = False
    hard_boundary_after: bool = False

    def __post_init__(self) -> None:
        if (
            not self.text
            or self.analysis_start < 0
            or self.analysis_end <= self.analysis_start
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("invalid aligned analysis token")
        source_values = (self.source_start, self.source_end)
        if any(value is None for value in source_values) and any(
            value is not None for value in source_values
        ):
            raise ValueError("aligned source offsets must be supplied together")
        if self.source_start is not None and (
            self.source_start < 0 or self.source_end is None or self.source_end <= self.source_start
        ):
            raise ValueError("invalid aligned source offsets")

    @property
    def display_start(self) -> int | None:
        """Alias for the immutable display/source start offset."""
        return self.source_start

    @property
    def display_end(self) -> int | None:
        """Alias for the immutable display/source end offset."""
        return self.source_end


@dataclass(frozen=True, slots=True)
class AnalysisNormalization:
    """An analysis string and its immutable, display-aligned lexical metadata."""

    display_text: str
    analysis_text: str
    tokens: tuple[AlignedToken, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.display_text, str) or not isinstance(self.analysis_text, str):
            raise TypeError("analysis normalization text values must be strings")
        prior_analysis_end = 0
        prior_source_end = 0
        for token in self.tokens:
            if token.analysis_start < prior_analysis_end or token.analysis_end > len(
                self.analysis_text
            ):
                raise ValueError("aligned analysis tokens must be ordered and non-overlapping")
            gap = self.analysis_text[prior_analysis_end : token.analysis_start]
            if gap and not gap.isspace():
                raise ValueError("analysis text contains unaligned non-whitespace content")
            if self.analysis_text[token.analysis_start : token.analysis_end] != token.text:
                raise ValueError("aligned token text does not match its analysis span")
            if token.source_start is None:
                if token.transformation != "filler_comma":
                    raise ValueError("only an inserted filler boundary may omit source offsets")
                if not token.hard_boundary_before or not token.hard_boundary_after:
                    raise ValueError("inserted filler punctuation must be a hard boundary")
            elif token.source_end is None or token.source_end > len(self.display_text):
                raise ValueError("aligned token source span is outside display text")
            elif token.source_start < prior_source_end:
                raise ValueError("aligned source spans must be ordered and non-overlapping")
            else:
                prior_source_end = token.source_end
            prior_analysis_end = token.analysis_end
        trailing = self.analysis_text[prior_analysis_end:]
        if trailing and not trailing.isspace():
            raise ValueError("analysis text contains trailing unaligned content")

    @property
    def text(self) -> str:
        """Compatibility-friendly shorthand for the model-facing text."""
        return self.analysis_text
