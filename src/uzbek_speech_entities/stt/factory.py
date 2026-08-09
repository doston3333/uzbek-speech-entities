"""Configuration-backed construction of the local Uzbek STT service."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import AppConfig, load_config, resolve_project_path
from .base import validate_immutable_revision
from .transformers_backend import TransformersSpeechToTextService


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"STT configuration field {name} must be a non-empty string.")
    return value


def _positive_number(values: Mapping[str, Any], name: str) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"STT configuration field {name} must be positive.")
    return float(value)


def _batch_size(values: Mapping[str, Any]) -> int:
    value = values.get("batch_size")
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("STT configuration batch_size must be exactly one.")
    return value


def _device_preference(values: Mapping[str, Any]) -> tuple[str, ...]:
    value = values.get("device_preference")
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("STT device_preference must be a non-empty sequence.")
    preference = tuple(value)
    if not preference or any(item not in {"mps", "cpu"} for item in preference):
        raise ValueError("STT device_preference must contain only mps or cpu.")
    return preference


def _environment_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")
    return value.strip()


def create_stt_service(
    config: AppConfig | None = None,
    *,
    use_fallback_model: bool = False,
    local_files_only: bool = False,
) -> TransformersSpeechToTextService:
    """Build STT from immutable app configuration and documented environment overrides.

    Selecting the base model is an explicit caller decision; a model-load error
    never causes an automatic fallback from the configured small model.
    """
    app_config = config or load_config()
    stt = app_config.section("stt")
    selected_key = "fallback_model_id" if use_fallback_model else "model_id"
    configured_model_id = _required_string(stt, selected_key)
    configured_revision = validate_immutable_revision(
        stt.get("fallback_model_revision" if use_fallback_model else "model_revision"),
        field_name=("stt.fallback_model_revision" if use_fallback_model else "stt.model_revision"),
    )
    environment_model_id = _environment_value("STT_MODEL_ID")
    environment_revision = _environment_value("STT_MODEL_REVISION")
    if environment_model_id is not None and environment_revision is None:
        raise ValueError("STT_MODEL_ID requires STT_MODEL_REVISION.")
    model_id = environment_model_id or configured_model_id
    revision = validate_immutable_revision(
        environment_revision or configured_revision,
        field_name="STT_MODEL_REVISION" if environment_revision else "STT revision",
    )
    cache_value = os.getenv("MODEL_CACHE_DIR", "./models/cache")
    if not cache_value.strip():
        raise ValueError("MODEL_CACHE_DIR must not be empty.")
    return TransformersSpeechToTextService(
        model_id=model_id,
        cache_dir=resolve_project_path(Path(cache_value)),
        revision=revision,
        language=_required_string(stt, "language"),
        task=_required_string(stt, "task"),
        chunk_length_seconds=_positive_number(stt, "chunk_length_seconds"),
        batch_size=_batch_size(stt),
        device_preference=_device_preference(stt),
        local_files_only=local_files_only,
    )
