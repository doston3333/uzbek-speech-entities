"""Speech-to-text interfaces and the local Transformers backend."""

from .base import ModelLoadError, SpeechToTextService, TranscriptionError
from .factory import create_stt_service
from .transformers_backend import TransformersSpeechToTextService

__all__ = [
    "ModelLoadError",
    "SpeechToTextService",
    "TranscriptionError",
    "TransformersSpeechToTextService",
    "create_stt_service",
]
