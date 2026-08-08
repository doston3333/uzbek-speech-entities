"""Canonical 16 kHz mono WAV preparation for speech recognition."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

from .loader import DecodedAudio
from .validation import AudioValidationConfig, AudioValidationError, validate_audio


@dataclass(frozen=True)
class ProcessedAudio:
    """Immutable, finite mono float32 samples ready for Whisper."""

    samples: NDArray[np.float32]
    sample_rate: int

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            raise ValueError("Processed audio sample rate must be positive.")
        samples = np.asarray(self.samples, dtype=np.float32).reshape(-1).copy()
        if not np.isfinite(samples).all():
            raise ValueError("Processed audio samples must be finite.")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def frames(self) -> int:
        """Return the number of mono frames."""
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        """Return the duration in seconds."""
        return self.frames / self.sample_rate


def preprocess_audio(decoded: DecodedAudio, target_sample_rate: int = 16_000) -> ProcessedAudio:
    """Downmix, resample, and safely scale decoded samples for Whisper."""
    if isinstance(target_sample_rate, bool) or target_sample_rate <= 0:
        raise AudioValidationError("Target sample rate must be positive.")
    if not np.isfinite(decoded.samples).all():
        raise AudioValidationError("Decoded audio samples must be finite.")

    mono = decoded.samples if decoded.samples.ndim == 1 else decoded.samples.mean(axis=1)
    samples = np.asarray(mono, dtype=np.float32)
    if decoded.sample_rate != target_sample_rate:
        samples = _resample(samples, decoded.sample_rate, target_sample_rate)
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.isfinite(samples).all():
        raise AudioValidationError("Preprocessed audio samples must be finite.")
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak
    if not np.isfinite(samples).all():
        raise AudioValidationError("Preprocessed audio samples must be finite.")
    return ProcessedAudio(samples=samples, sample_rate=target_sample_rate)


def _resample(
    samples: NDArray[np.float32], original_sample_rate: int, target_sample_rate: int
) -> NDArray[np.float32]:
    """Resample lazily so importing the audio API remains lightweight."""
    import librosa

    return np.asarray(
        librosa.resample(samples, orig_sr=original_sample_rate, target_sr=target_sample_rate),
        dtype=np.float32,
    )


def _can_reuse_original(path: Path, decoded: DecodedAudio, target_sample_rate: int) -> bool:
    """Return whether *path* is already canonical with no scaling required."""
    peak = float(np.max(np.abs(decoded.samples))) if decoded.samples.size else 0.0
    return (
        path.suffix.lower() == ".wav"
        and decoded.samples.ndim == 1
        and decoded.sample_rate == target_sample_rate
        and peak <= 1.0
    )


@contextmanager
def prepared_audio(path: Path, config: AudioValidationConfig) -> Iterator[Path]:
    """Yield a canonical WAV path, deleting a generated temporary file on exit."""
    source_path = Path(path)
    decoded = validate_audio(source_path, config)
    if _can_reuse_original(source_path, decoded, config.target_sample_rate):
        yield source_path
        return

    processed = preprocess_audio(decoded, config.target_sample_rate)
    temporary = NamedTemporaryFile(prefix="uzbek-stt-", suffix=".wav", delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        sf.write(temporary_path, processed.samples, processed.sample_rate, subtype="FLOAT")
        yield temporary_path
    finally:
        temporary_path.unlink(missing_ok=True)


prepare_audio = prepared_audio
