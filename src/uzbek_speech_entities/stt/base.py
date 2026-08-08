"""Small typed interface shared by speech-to-text implementations."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


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
