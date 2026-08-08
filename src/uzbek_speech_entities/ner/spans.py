"""BIO aggregation and invariant checking for public entity spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from .schemas import Entity, PublicEntityLabel


class NERPredictionError(RuntimeError):
    """Raised when a NER prediction cannot be decoded into safe public spans."""


@dataclass(frozen=True)
class TokenPrediction:
    """The highest-probability model label for one lexical unit."""

    label: str
    score: float
    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            not self.label
            or not 0.0 <= self.score <= 1.0
            or self.start < 0
            or self.end <= self.start
        ):
            raise NERPredictionError("Invalid token prediction.")


def validate_entity_spans(text: str, entities: Sequence[Entity]) -> tuple[Entity, ...]:
    """Ensure entities are ordered, non-overlapping, and exact source slices."""
    previous_end = 0
    for entity in entities:
        if entity.end > len(text) or entity.start < previous_end:
            raise NERPredictionError("NER returned invalid or overlapping entity offsets.")
        if text[entity.start : entity.end] != entity.text:
            raise NERPredictionError("NER entity text does not match its source span.")
        previous_end = entity.end
    return tuple(entities)


def aggregate_bio_predictions(
    text: str,
    predictions: Sequence[TokenPrediction],
    *,
    model_to_application_labels: Mapping[str, str],
    visible_labels: frozenset[str],
    threshold: float,
) -> tuple[Entity, ...]:
    """Merge lexical-unit BIO tags, map labels, and return validated public spans.

    An orphan or incompatible ``I-X`` starts a new entity.  Grouping happens
    before thresholding so an entity score is the mean of all member tokens.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("NER confidence threshold must be in [0, 1].")
    ordered = sorted(predictions, key=lambda item: (item.start, item.end))
    groups: list[list[TokenPrediction]] = []
    current: list[TokenPrediction] = []
    current_type: str | None = None

    def flush() -> None:
        nonlocal current, current_type
        if current:
            groups.append(current)
        current = []
        current_type = None

    for prediction in ordered:
        if prediction.end > len(text):
            raise NERPredictionError("NER token offset exceeds source text.")
        label = prediction.label
        if label == "O":
            flush()
            continue
        try:
            prefix, entity_type = label.split("-", maxsplit=1)
        except ValueError:
            flush()
            continue
        if prefix not in {"B", "I"} or not entity_type:
            flush()
            continue
        if prefix == "I" and current and current_type == entity_type:
            current.append(prediction)
            continue
        flush()
        current = [prediction]
        current_type = entity_type
    flush()

    candidates: list[Entity] = []
    for group in groups:
        mapped_label = model_to_application_labels.get(group[0].label.split("-", 1)[1])
        if mapped_label not in visible_labels or mapped_label not in {"PER", "LOC", "ORG", "DATE"}:
            continue
        application_label = cast(PublicEntityLabel, mapped_label)
        score = sum(member.score for member in group) / len(group)
        if score < threshold:
            continue
        start, end = group[0].start, group[-1].end
        candidates.append(
            Entity(
                text=text[start:end], label=application_label, start=start, end=end, score=score
            )
        )

    # Prefer confidence, then earlier/longer spans, then a stable label tie-break.
    kept: list[Entity] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            -(item.score if item.score is not None else 0.0),
            item.start,
            -item.end,
            item.label,
        ),
    ):
        if all(candidate.end <= chosen.start or candidate.start >= chosen.end for chosen in kept):
            kept.append(candidate)
    return validate_entity_spans(text, sorted(kept, key=lambda item: (item.start, item.end)))
