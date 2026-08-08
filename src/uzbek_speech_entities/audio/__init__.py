"""Audio validation, decoding, and canonicalization interfaces."""

from .loader import AudioDecodingError, DecodedAudio, decode_audio
from .preprocessing import ProcessedAudio, prepare_audio, prepared_audio, preprocess_audio
from .validation import (
    AudioValidationConfig,
    AudioValidationError,
    validate_audio,
    validate_audio_file,
)

__all__ = [
    "AudioDecodingError",
    "AudioValidationConfig",
    "AudioValidationError",
    "DecodedAudio",
    "ProcessedAudio",
    "decode_audio",
    "prepare_audio",
    "prepared_audio",
    "preprocess_audio",
    "validate_audio",
    "validate_audio_file",
]
