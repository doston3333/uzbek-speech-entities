"""Lazy Transformers implementation of Uzbek Whisper transcription."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from ..audio.loader import AudioDecodingError, decode_audio
from .base import ModelLoadError, TranscriptionError

LOGGER = logging.getLogger(__name__)
_DEFAULT_NATIVE_CHUNK_SECONDS = 30.0
_ENERGY_WINDOW_SECONDS = 0.25
_ENERGY_HOP_SECONDS = 0.05
_MINIMUM_FINAL_CHUNK_SECONDS = 3.0


def _split_audio_on_quiet_boundaries(
    samples: np.ndarray,
    sample_rate: int,
    max_chunk_seconds: float,
) -> tuple[np.ndarray, ...]:
    """Split mono audio into bounded chunks, preferring the earliest strong pause."""
    max_frames = max(1, int(round(max_chunk_seconds * sample_rate)))
    if samples.size <= max_frames:
        return (samples,)

    window_frames = max(1, int(round(_ENERGY_WINDOW_SECONDS * sample_rate)))
    hop_frames = max(1, int(round(_ENERGY_HOP_SECONDS * sample_rate)))
    minimum_tail = max(1, int(round(_MINIMUM_FINAL_CHUNK_SECONDS * sample_rate)))
    chunks: list[np.ndarray] = []
    start = 0

    while samples.size - start > max_frames:
        earliest = start + max_frames // 2
        latest = min(
            start + max_frames - window_frames // 2,
            samples.size - minimum_tail,
        )
        if latest <= earliest:
            cut = min(start + max_frames, samples.size)
        else:
            half_window = window_frames // 2
            candidates: list[tuple[float, int]] = []
            for center in range(earliest, latest + 1, hop_frames):
                window_start = max(start, center - half_window)
                window_end = min(samples.size, window_start + window_frames)
                window = samples[window_start:window_end].astype(np.float64, copy=False)
                rms = float(np.sqrt(np.mean(np.square(window))))
                candidates.append((rms, center))

            median_energy = float(np.median([energy for energy, _ in candidates]))
            quiet_threshold = max(1e-4, median_energy * 0.25)
            strong_pauses = [
                center for energy, center in candidates if energy <= quiet_threshold
            ]
            if strong_pauses:
                cut = strong_pauses[0]
            else:
                late_start = start + int(max_frames * 0.7)
                late_candidates = [
                    candidate for candidate in candidates if candidate[1] >= late_start
                ]
                search_space = late_candidates or candidates
                _, cut = min(search_space, key=lambda candidate: (candidate[0], -candidate[1]))

        if cut <= start:
            cut = min(start + max_frames, samples.size)
        chunks.append(samples[start:cut])
        start = cut

    if start < samples.size:
        chunks.append(samples[start:])
    return tuple(chunks)


class TransformersSpeechToTextService:
    """Load one Whisper model lazily and transcribe canonical audio.

    Long recordings are split at quiet boundaries and each bounded segment uses
    the checkpoint's reliable short-form, non-timestamp generation path.
    """

    def __init__(
        self,
        model_id: str,
        cache_dir: Path,
        language: str = "uz",
        task: str = "transcribe",
        chunk_length_seconds: float = 30.0,
        batch_size: int = 1,
        device_preference: tuple[str, ...] = ("mps", "cpu"),
        *,
        local_files_only: bool = False,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("STT model ID must be a non-empty string.")
        if (
            not isinstance(language, str)
            or not language.strip()
            or not isinstance(task, str)
            or not task.strip()
        ):
            raise ValueError("STT language and task must be non-empty strings.")
        if isinstance(chunk_length_seconds, bool) or chunk_length_seconds <= 0:
            raise ValueError("STT chunk length must be positive.")
        if isinstance(batch_size, bool) or batch_size != 1:
            raise ValueError("STT batch size must be exactly one.")
        if not device_preference or any(
            device not in {"mps", "cpu"} for device in device_preference
        ):
            raise ValueError("STT device preferences must contain only mps or cpu.")
        self._model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.language = language
        self.task = task
        self.chunk_length_seconds = float(chunk_length_seconds)
        self.batch_size = batch_size
        self.device_preference = tuple(device_preference)
        self.local_files_only = local_files_only
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._load_error: ModelLoadError | None = None
        self._load_attempted = False
        self._load_lock = RLock()

    @property
    def model_id(self) -> str:
        """Return the configured Hugging Face model identifier."""
        return self._model_id

    @property
    def loaded(self) -> bool:
        """Return whether model loading has completed successfully."""
        return self._model is not None and self._processor is not None

    @property
    def device(self) -> str | None:
        """Return the selected device after model loading."""
        return self._device

    def load(self) -> None:
        """Eagerly initialize the existing one-shot model loader."""
        self._ensure_loaded()

    def _load_dependencies(self) -> tuple[Any, Any, Any]:
        """Import optional ML dependencies only when a model is requested."""
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        return torch, AutoProcessor, AutoModelForSpeechSeq2Seq

    def _select_device(self, torch_module: Any) -> str:
        mps_available = bool(torch_module.backends.mps.is_available())
        for candidate in self.device_preference:
            if candidate == "mps" and mps_available:
                return "mps"
            if candidate == "cpu":
                if not mps_available:
                    LOGGER.warning("Using CPU for STT because MPS is unavailable.")
                return "cpu"
        raise ModelLoadError("No configured STT device is available.")

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            if self._load_error is not None:
                raise self._load_error
            return
        with self._load_lock:
            if self._load_attempted:
                if self._load_error is not None:
                    raise self._load_error
                return
            self._load_attempted = True
            try:
                torch_module, processor_class, model_class = self._load_dependencies()
                device = self._select_device(torch_module)
                processor = processor_class.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir),
                    local_files_only=self.local_files_only,
                )
                model = model_class.from_pretrained(
                    self.model_id,
                    cache_dir=str(self.cache_dir),
                    dtype=torch_module.float32,
                    local_files_only=self.local_files_only,
                )
                model.to(device)
                model.eval()
                self._processor = processor
                self._model = model
                self._torch = torch_module
                self._device = device
                LOGGER.info("Loaded STT model %s on %s.", self.model_id, device)
            except ModelLoadError as error:
                self._load_error = error
                raise
            except Exception as error:
                self._load_error = ModelLoadError("STT model could not be loaded.")
                raise self._load_error from error

    def _maximum_chunk_seconds(self) -> float:
        if self._processor is None:
            raise TranscriptionError("STT processor is unavailable.")
        configured_native_chunk = getattr(
            self._processor.feature_extractor,
            "chunk_length",
            _DEFAULT_NATIVE_CHUNK_SECONDS,
        )
        try:
            native_chunk_seconds = float(configured_native_chunk)
        except (TypeError, ValueError):
            native_chunk_seconds = _DEFAULT_NATIVE_CHUNK_SECONDS
        if not np.isfinite(native_chunk_seconds) or native_chunk_seconds <= 0:
            native_chunk_seconds = _DEFAULT_NATIVE_CHUNK_SECONDS
        return min(self.chunk_length_seconds, native_chunk_seconds)

    def _transcribe_chunk(self, samples: np.ndarray) -> str:
        if (
            self._processor is None
            or self._model is None
            or self._torch is None
            or self._device is None
        ):
            raise TranscriptionError("STT model is unavailable.")
        inputs = self._processor(
            samples,
            sampling_rate=16_000,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            return_attention_mask=True,
        )
        inputs = inputs.to(self._device, self._torch.float32)
        with self._torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                language=self.language,
                task=self.task,
                return_timestamps=False,
                use_model_defaults=False,
            )
        transcripts = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        transcript = transcripts[0] if isinstance(transcripts, list) and transcripts else None
        if not isinstance(transcript, str):
            raise TranscriptionError("STT model returned an invalid transcript.")
        return transcript.strip()

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe a finite 16 kHz mono audio file without exposing its content."""
        self._ensure_loaded()
        try:
            decoded = decode_audio(Path(audio_path))
            if decoded.sample_rate != 16_000 or not decoded.is_mono:
                raise TranscriptionError("STT input must be mono 16 kHz audio.")
            if decoded.frames <= 0 or not np.isfinite(decoded.samples).all():
                raise TranscriptionError("STT input contains invalid audio samples.")
            chunks = _split_audio_on_quiet_boundaries(
                decoded.samples,
                decoded.sample_rate,
                self._maximum_chunk_seconds(),
            )
            transcripts = [self._transcribe_chunk(chunk) for chunk in chunks]
            return " ".join(transcript for transcript in transcripts if transcript)
        except (AudioDecodingError, TranscriptionError):
            raise
        except Exception as error:
            raise TranscriptionError("Audio transcription could not be completed.") from error
