"""Audio upload validation that decodes an accepted file exactly once."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .loader import AudioDurationLimitError, DecodedAudio, decode_audio


class AudioValidationError(ValueError):
    """Raised when an upload violates the configured audio policy."""


@dataclass(frozen=True)
class AudioValidationConfig:
    """Immutable, validated values from the application's ``audio`` mapping."""

    target_sample_rate: int
    mono: bool
    max_seconds: float
    max_upload_bytes: int
    allowed_extensions: frozenset[str]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> AudioValidationConfig:
        """Parse the existing YAML audio mapping without mutating it."""
        target_sample_rate = values.get("target_sample_rate")
        mono = values.get("mono")
        max_seconds = values.get("max_seconds")
        max_upload_mb = values.get("max_upload_mb")
        extensions = values.get("allowed_extensions")
        if isinstance(target_sample_rate, bool) or not isinstance(target_sample_rate, int):
            raise AudioValidationError("Audio target sample rate must be a positive integer.")
        if target_sample_rate <= 0 or not isinstance(mono, bool):
            raise AudioValidationError("Audio configuration contains an invalid sample policy.")
        if isinstance(max_seconds, bool) or not isinstance(max_seconds, int | float):
            raise AudioValidationError("Audio maximum duration must be positive.")
        if isinstance(max_upload_mb, bool) or not isinstance(max_upload_mb, int | float):
            raise AudioValidationError("Audio maximum upload size must be positive.")
        if float(max_seconds) <= 0 or float(max_upload_mb) <= 0:
            raise AudioValidationError("Audio limits must be positive.")
        if isinstance(extensions, str | bytes) or not isinstance(extensions, Sequence):
            raise AudioValidationError("Audio extensions must be a non-empty sequence.")
        normalized_extensions = frozenset(
            item.strip().lower().lstrip(".")
            for item in extensions
            if isinstance(item, str) and item.strip()
        )
        if len(normalized_extensions) != len(extensions) or not normalized_extensions:
            raise AudioValidationError("Audio extensions must be non-empty strings.")
        return cls(
            target_sample_rate=target_sample_rate,
            mono=mono,
            max_seconds=float(max_seconds),
            max_upload_bytes=int(float(max_upload_mb) * 1024 * 1024),
            allowed_extensions=normalized_extensions,
        )


def validate_audio(path: Path, config: AudioValidationConfig) -> DecodedAudio:
    """Validate an upload and return its already-decoded samples.

    ``AudioDecodingError`` intentionally remains distinct so callers can map it
    to the application's typed decoding failure rather than a generic policy
    rejection.
    """
    candidate = Path(path)
    if not candidate.exists():
        raise AudioValidationError("Audio file does not exist.")
    if not candidate.is_file():
        raise AudioValidationError("Audio path must be a regular file.")
    if candidate.stat().st_size <= 0:
        raise AudioValidationError("Audio file is empty.")
    extension = candidate.suffix.lower().lstrip(".")
    if extension not in config.allowed_extensions:
        raise AudioValidationError("Audio file format is not allowed.")
    if candidate.stat().st_size > config.max_upload_bytes:
        raise AudioValidationError("Audio file exceeds the upload size limit.")

    try:
        decoded = decode_audio(candidate, max_seconds=config.max_seconds)
    except AudioDurationLimitError as error:
        raise AudioValidationError("Audio exceeds the duration limit.") from error
    if decoded.frames <= 0:
        raise AudioValidationError("Decoded audio contains no frames.")
    if decoded.sample_rate <= 0:
        raise AudioValidationError("Decoded audio has an invalid sample rate.")
    if not np.isfinite(decoded.samples).all():
        raise AudioValidationError("Decoded audio contains non-finite samples.")
    if decoded.duration_seconds > config.max_seconds:
        raise AudioValidationError("Audio exceeds the duration limit.")
    return decoded


def validate_audio_file(path: Path, config: AudioValidationConfig) -> DecodedAudio:
    """Compatibility spelling for :func:`validate_audio`."""
    return validate_audio(path, config)
