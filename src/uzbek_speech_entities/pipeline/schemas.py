"""Stable pipeline result schemas shared by text and audio analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..ner.schemas import Entity


class Timing(BaseModel):
    """Per-stage monotonic timings measured in milliseconds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_preprocessing_ms: float = Field(ge=0.0)
    stt_ms: float = Field(ge=0.0)
    normalization_ms: float = Field(ge=0.0)
    ner_ms: float = Field(ge=0.0)
    total_ms: float = Field(ge=0.0)


class AnalysisModels(BaseModel):
    """The local model identities used for an analysis response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stt: str | None = None
    stt_revision: str | None = None
    ner: str


class AnalysisResult(BaseModel):
    """The uniform API-independent result of text or audio analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_transcript: str
    normalized_transcript: str
    entities: tuple[Entity, ...]
    timing: Timing
    models: AnalysisModels
