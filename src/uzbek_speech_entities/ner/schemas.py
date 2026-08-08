"""Public, validated named-entity schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

PublicEntityLabel = Literal["PER", "LOC", "ORG", "DATE"]
EntitySource = Literal[
    "temporal_grammar",
    "person_gazetteer",
    "gazetteer_boundary_expansion",
    "person_introduction",
    "model_boundary_expansion",
    "clean_model",
    "normalized_clean_model",
    "person_relation",
]
CanonicalSource = Literal["name_lexicon_edit_distance", "person_phrase_gazetteer"]


class Entity(BaseModel):
    """One non-empty public entity span in a normalized transcript."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str = Field(min_length=1)
    label: PublicEntityLabel
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    source: EntitySource = "clean_model"
    evidence: tuple[str, ...] | None = None
    canonical_text: str | None = Field(default=None, min_length=1)
    canonical_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    canonical_source: CanonicalSource | None = None

    @model_validator(mode="after")
    def _end_follows_start(self) -> Entity:
        if self.end <= self.start:
            raise ValueError("entity end must be greater than start")
        canonical_values = (
            self.canonical_text,
            self.canonical_confidence,
            self.canonical_source,
        )
        if any(value is not None for value in canonical_values) and not all(
            value is not None for value in canonical_values
        ):
            raise ValueError("canonical entity metadata must be supplied together")
        if self.label != "PER" and self.canonical_text is not None:
            raise ValueError("canonical person metadata is valid only for PER entities")
        return self
