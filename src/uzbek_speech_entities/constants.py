"""Immutable application-wide constants with no runtime side effects."""

from typing import Final

APPLICATION_LABELS: Final[tuple[str, ...]] = ("PER", "LOC", "ORG", "DATE")
CANONICAL_APOSTROPHE: Final[str] = "ʻ"
DIAGNOSTIC_TRANSCRIPT_LOGGING_ENV: Final[str] = "DIAGNOSTIC_TRANSCRIPT_LOGGING"
