"""Model-anchored ORG/LOC completion for split speech entities."""

from __future__ import annotations

from typing import Literal, cast

from ..offset_tokens import WordToken
from ..schemas import Entity
from ..span_resolver import Candidate
from .resources import load_resource

_HEAD_ENDINGS = frozenset(
    {
        "",
        "i",
        "ida",
        "iga",
        "idan",
        "si",
        "sida",
        "siga",
        "sidan",
        "da",
        "dan",
        "ga",
        "ka",
        "ni",
        "ning",
        "lari",
        "larida",
    }
)


def _head_bases(head: str) -> frozenset[str]:
    bases = {head}
    if head.endswith(("k", "q")):
        bases.add(f"{head[:-1]}g")
    if head == "shahar":
        bases.add("shahr")
    return frozenset(bases)


def semantic_head_label(token: WordToken) -> Literal["ORG", "LOC"] | None:
    resource = load_resource("heads_stopwords.json")
    key = token.comparison_key
    for label, field in (("ORG", "organization_heads"), ("LOC", "location_heads")):
        for head in resource[field]:
            for base in _head_bases(head):
                if key.startswith(base) and key[len(base) :] in _HEAD_ENDINGS:
                    return cast(Literal["ORG", "LOC"], label)
    return None


def boundary_expansion_candidates(
    tokens: tuple[WordToken, ...], model_entities: tuple[Entity, ...]
) -> tuple[Candidate, ...]:
    """Expand only a model ORG/LOC anchor to a nearby semantic head."""
    terms = load_resource("heads_stopwords.json")
    safe_modifiers = frozenset(terms["safe_modifiers"])
    blocked = frozenset(terms["boundary_stopwords"])
    temporal_terms = load_resource("temporal_terms.json")
    temporal_blockers = frozenset(
        item
        for field in ("months", "weekdays", "relative_dates", "relative_modifiers", "periods")
        for item in temporal_terms[field]
    )

    def is_temporal_blocker(key: str) -> bool:
        return key in temporal_blockers or any(
            key.startswith(item) and key[len(item) :] in _HEAD_ENDINGS for item in temporal_blockers
        )

    candidates: list[Candidate] = []
    for entity in model_entities:
        if entity.label not in {"ORG", "LOC"}:
            continue
        anchor_index = next(
            (
                index
                for index, token in enumerate(tokens)
                if token.start == entity.start and token.end <= entity.end
            ),
            None,
        )
        if anchor_index is None:
            continue
        head_index: int | None = None
        label: Literal["ORG", "LOC"] | None = None
        for index in range(anchor_index + 1, min(len(tokens), anchor_index + 4)):
            if (
                tokens[index].boundary_before
                or tokens[index].comparison_key in blocked
                or is_temporal_blocker(tokens[index].comparison_key)
            ):
                break
            label = semantic_head_label(tokens[index])
            if label is not None:
                head_index = index
                break
        if head_index is None or label is None:
            continue
        start = entity.start
        if (
            anchor_index > 0
            and not tokens[anchor_index].boundary_before
            and tokens[anchor_index - 1].comparison_key in safe_modifiers
        ):
            start = tokens[anchor_index - 1].start
        candidates.append(
            Candidate(
                label=label,  # semantic head deliberately retypes the model anchor
                start=start,
                end=tokens[head_index].end,
                source="model_boundary_expansion",
                evidence=("model_anchor_semantic_head",),
            )
        )
    return tuple(candidates)
