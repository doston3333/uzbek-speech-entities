"""Named-entity inference and safe public entity spans."""

from .predictor import NERPredictor, NERService
from .schemas import Entity
from .spans import (
    NERPredictionError,
    TokenPrediction,
    aggregate_bio_predictions,
    validate_entity_spans,
)

__all__ = [
    "Entity",
    "NERPredictionError",
    "NERPredictor",
    "NERService",
    "TokenPrediction",
    "aggregate_bio_predictions",
    "validate_entity_spans",
]
