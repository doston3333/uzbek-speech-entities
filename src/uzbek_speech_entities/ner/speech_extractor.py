"""Opt-in precision-first rescue for normalized audio transcripts."""

from __future__ import annotations

from .canonicalize_person import canonicalize_person_entities
from .offset_tokens import tokenize_words
from .rules.boundaries import boundary_expansion_candidates
from .rules.person import person_introduction_candidates, person_relation_candidates
from .rules.person_gazetteer import (
    gazetteer_boundary_expansion_candidates,
    person_gazetteer_candidates,
)
from .rules.temporal import temporal_candidates
from .schemas import Entity
from .span_resolver import Candidate, resolve_candidates


class SpeechNERRescue:
    """Combine deterministic speech rules with the one clean-model prediction."""

    def extract(
        self,
        text: str,
        model_entities: tuple[Entity, ...],
        normalized_model_entities: tuple[Candidate, ...] = (),
    ) -> tuple[Entity, ...]:
        tokens = tokenize_words(text)
        clean = tuple(
            Candidate(
                label=entity.label,
                start=entity.start,
                end=entity.end,
                source="clean_model",
                score=entity.score,
                evidence=entity.evidence or ("clean_model",),
            )
            for entity in model_entities
        )
        candidates = (
            *temporal_candidates(tokens),
            *gazetteer_boundary_expansion_candidates(tokens),
            *person_gazetteer_candidates(tokens),
            *person_introduction_candidates(tokens),
            *boundary_expansion_candidates(tokens, model_entities),
            *clean,
            *normalized_model_entities,
            *person_relation_candidates(tokens),
        )
        return canonicalize_person_entities(resolve_candidates(text, candidates))
