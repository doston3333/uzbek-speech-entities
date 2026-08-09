"""Small typed interface shared by speech-to-text implementations."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

IMMUTABLE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")


def validate_immutable_revision(value: object, *, field_name: str = "STT revision") -> str:
    """Return a full lowercase Git commit hash or fail before model resolution."""
    if not isinstance(value, str) or not IMMUTABLE_REVISION_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase 40-character Git commit hash.")
    return value


class ModelLoadError(RuntimeError):
    """Raised when an STT model cannot be loaded."""


class TranscriptionError(RuntimeError):
    """Raised when transcription cannot be completed safely."""


@runtime_checkable
class SpeechToTextService(Protocol):
    """A service that transcribes one canonical audio path at a time."""

    @property
    def model_id(self) -> str:
        """Identifier of the loaded speech model."""
        ...

    @property
    def revision(self) -> str:
        """Immutable Git revision of the loaded speech model."""
        ...

    @property
    def loaded(self) -> bool:
        """Whether the model has completed loading successfully."""
        ...

    @property
    def device(self) -> str | None:
        """Selected inference device, once loaded."""
        ...

    def load(self) -> None:
        """Eagerly load the service model exactly once."""
        ...

    def transcribe(self, audio_path: Path) -> str:
        """Return a plain transcript for ``audio_path``."""
        ...
