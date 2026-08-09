"""Audio decoding primitives used by validation and STT preprocessing."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class AudioDecodingError(RuntimeError):
    """Raised when an audio file cannot be decoded safely."""


class AudioDurationLimitError(AudioDecodingError):
    """Raised when bounded decoding proves that audio exceeds the policy limit."""


@dataclass(frozen=True)
class DecodedAudio:
    """Decoded float32 samples while preserving the source mono/stereo layout."""

    samples: NDArray[np.float32]
    sample_rate: int

    def __post_init__(self) -> None:
        if isinstance(self.sample_rate, bool) or self.sample_rate <= 0:
            raise ValueError("Audio sample rate must be positive.")
        samples = np.asarray(self.samples, dtype=np.float32).copy()
        if samples.ndim not in (1, 2):
            raise ValueError("Audio samples must be mono or channel-last multi-channel data.")
        samples.setflags(write=False)
        object.__setattr__(self, "samples", samples)

    @property
    def frames(self) -> int:
        """Return the number of sample frames."""
        return int(self.samples.shape[0])

    @property
    def channels(self) -> int:
        """Return the number of channels in the decoded source."""
        return 1 if self.samples.ndim == 1 else int(self.samples.shape[1])

    @property
    def is_mono(self) -> bool:
        """Whether the decoded representation has one channel."""
        return self.channels == 1

    @property
    def duration_seconds(self) -> float:
        """Return the decoded duration in seconds."""
        return self.frames / self.sample_rate


def decode_audio(path: Path, *, max_seconds: float | None = None) -> DecodedAudio:
    """Decode *path* without changing channel layout or sample rate.

    SoundFile is the fast path for WAV, FLAC, and MP3.  The configured
    browser/upload formats also include M4A and WebM, which are not exposed by
    every libsndfile build, so FFmpeg decodes unsupported containers to an
    in-memory float WAV.  When ``max_seconds`` is supplied, both paths cap the
    number of decoded frames before materializing the complete upload.  No
    temporary file or shell invocation is used.
    """
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, int | float)
        or not math.isfinite(float(max_seconds))
        or float(max_seconds) <= 0
    ):
        raise ValueError("Audio decoding duration limit must be positive and finite.")
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        frame_limit = -1
        if max_seconds is not None:
            info = sf.info(path)
            if info.samplerate <= 0:
                raise RuntimeError("Audio sample rate is invalid.")
            frame_limit = math.floor(float(max_seconds) * info.samplerate) + 1
        samples, sample_rate = sf.read(
            path,
            dtype="float32",
            always_2d=False,
            frames=frame_limit,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        try:
            samples, sample_rate = _decode_with_ffmpeg(path, max_seconds=max_seconds)
        except (OSError, RuntimeError, TypeError, ValueError, subprocess.SubprocessError) as error:
            raise AudioDecodingError("Audio file could not be decoded.") from error
    try:
        decoded = DecodedAudio(samples=samples, sample_rate=int(sample_rate))
    except (TypeError, ValueError) as error:
        raise AudioDecodingError("Audio file could not be decoded.") from error
    if max_seconds is not None and decoded.duration_seconds > float(max_seconds):
        raise AudioDurationLimitError("Audio exceeds the duration limit.")
    return decoded


def _decode_with_ffmpeg(
    path: Path, *, max_seconds: float | None = None
) -> tuple[NDArray[np.float32], int]:
    """Decode one stream with bounded output when a duration limit is supplied."""
    output_options: list[str] = []
    if max_seconds is not None:
        # Decode at most one second beyond the policy threshold so validation
        # can distinguish an exact-limit file from an over-limit file.  Capping
        # channels and rate bounds captured stdout to about 24 MB at 60 seconds.
        output_options = [
            "-t",
            f"{float(max_seconds) + 1.0:.6f}",
            "-ar",
            "48000",
            "-ac",
            "2",
        ]
    completed = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            *output_options,
            "-c:a",
            "pcm_f32le",
            "-f",
            "wav",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    if not completed.stdout:
        raise RuntimeError("FFmpeg returned no decoded audio.")

    import soundfile as sf

    samples, sample_rate = sf.read(BytesIO(completed.stdout), dtype="float32", always_2d=False)
    return np.asarray(samples, dtype=np.float32), int(sample_rate)
