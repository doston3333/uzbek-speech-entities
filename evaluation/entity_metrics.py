"""Exact multiset named-entity scoring in span and STT-surface modes."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Final, Literal, Protocol, cast

from uzbek_speech_entities.normalization import (
    normalize_evaluation,
    normalize_runtime,
)

APPLICATION_LABELS: Final[tuple[str, ...]] = ("PER", "LOC", "ORG", "DATE")
MetricMode = Literal["span", "surface"]
_TOKEN_RE = re.compile(r"\w+(?:[ʻ’‘ʼ'`-]\w+)*", flags=re.UNICODE)


class EntityLike(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def start(self) -> int: ...

    @property
    def end(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ProjectedEntity:
    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _TokenSpan:
    normalized: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class EntityScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class EntityMetrics:
    by_label: dict[str, EntityScore]
    overall: EntityScore
    macro_f1: float


def _validate_entity(entity: object) -> EntityLike:
    text = getattr(entity, "text", None)
    label = getattr(entity, "label", None)
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    if not isinstance(text, str) or not text:
        raise ValueError("entity text must be a non-empty string")
    if label not in APPLICATION_LABELS:
        raise ValueError(f"entity label must be one of {APPLICATION_LABELS}")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise ValueError("entity offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError("entity offsets must form a non-empty non-negative span")
    return cast(EntityLike, entity)


def _key(entity: EntityLike, mode: MetricMode) -> tuple[str, int, int] | tuple[str, str]:
    if mode == "span":
        return (entity.label, entity.start, entity.end)
    normalized = normalize_evaluation(entity.text)
    if not normalized:
        raise ValueError("entity text must have a non-empty normalized surface")
    return (entity.label, normalized)


def _score(gold: Counter[object], predicted: Counter[object]) -> EntityScore:
    tp = sum((gold & predicted).values())
    support = sum(gold.values())
    predicted_count = sum(predicted.values())
    fp = predicted_count - tp
    fn = support - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EntityScore(tp, fp, fn, support, precision, recall, f1)


def calculate_entity_metrics(
    gold_entities: Iterable[object], predicted_entities: Iterable[object], *, mode: MetricMode
) -> EntityMetrics:
    """Calculate exact, duplicate-aware entity metrics for a declared coordinate mode."""
    if mode not in {"span", "surface"}:
        raise ValueError("mode must be 'span' or 'surface'")
    gold = [_validate_entity(entity) for entity in gold_entities]
    predicted = [_validate_entity(entity) for entity in predicted_entities]
    by_label: dict[str, EntityScore] = {}
    all_gold: Counter[object] = Counter()
    all_predicted: Counter[object] = Counter()
    for label in APPLICATION_LABELS:
        gold_keys: Counter[object] = Counter(
            _key(entity, mode) for entity in gold if entity.label == label
        )
        predicted_keys: Counter[object] = Counter(
            _key(entity, mode) for entity in predicted if entity.label == label
        )
        by_label[label] = _score(gold_keys, predicted_keys)
        all_gold.update(gold_keys)
        all_predicted.update(predicted_keys)
    overall = _score(all_gold, all_predicted)
    return EntityMetrics(
        by_label=by_label,
        overall=overall,
        macro_f1=sum(by_label[label].f1 for label in APPLICATION_LABELS) / len(APPLICATION_LABELS),
    )


def _token_spans(text: str) -> tuple[_TokenSpan, ...]:
    tokens: list[_TokenSpan] = []
    for match in _TOKEN_RE.finditer(text):
        normalized = normalize_evaluation(match.group())
        if normalized:
            tokens.append(_TokenSpan(normalized, match.start(), match.end()))
    return tuple(tokens)


def align_entities_to_reference(
    reference_text: str,
    hypothesis_text: str,
    entities: Iterable[object],
) -> tuple[ProjectedEntity, ...]:
    """Project hypothesis entities through exact token matches into reference offsets.

    An entity containing an inserted or substituted token receives a unique
    out-of-range sentinel span. It therefore counts as a false positive without
    ever being mistaken for an exact reference span. End-to-end exact-span F1
    stays strict: STT must preserve every entity token before NER receives credit.
    """
    if not isinstance(reference_text, str) or not isinstance(hypothesis_text, str):
        raise TypeError("reference_text and hypothesis_text must be strings")
    validated = [_validate_entity(entity) for entity in entities]
    reference_tokens = _token_spans(reference_text)
    hypothesis_tokens = _token_spans(hypothesis_text)
    matcher = SequenceMatcher(
        a=[token.normalized for token in hypothesis_tokens],
        b=[token.normalized for token in reference_tokens],
        autojunk=False,
    )
    token_mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            token_mapping[block.a + offset] = block.b + offset

    projected: list[ProjectedEntity] = []
    unmatched_start = len(reference_text) + 1
    for entity in validated:
        if (
            entity.end > len(hypothesis_text)
            or hypothesis_text[entity.start : entity.end] != entity.text
        ):
            raise ValueError("predicted entity span does not match hypothesis_text")
        contained = [
            index
            for index, token in enumerate(hypothesis_tokens)
            if token.start >= entity.start and token.end <= entity.end
        ]
        mapped = [token_mapping[index] for index in contained if index in token_mapping]
        exact_boundary = bool(contained) and (
            hypothesis_tokens[contained[0]].start == entity.start
            and hypothesis_tokens[contained[-1]].end == entity.end
        )
        contiguous = len(mapped) == len(contained) and all(
            current == previous + 1 for previous, current in zip(mapped, mapped[1:], strict=False)
        )
        if exact_boundary and contiguous:
            start = reference_tokens[mapped[0]].start
            end = reference_tokens[mapped[-1]].end
            text = reference_text[start:end]
        else:
            start = unmatched_start
            end = start + 1
            text = entity.text
            unmatched_start = end + 1
        projected.append(ProjectedEntity(text=text, label=entity.label, start=start, end=end))
    return tuple(projected)


def project_gold_entities(
    gold_transcript: str, entities: Iterable[object]
) -> tuple[ProjectedEntity, ...]:
    """Project validated original-text entity spans into ``normalize_runtime`` coordinates."""
    if not isinstance(gold_transcript, str):
        raise TypeError("gold_transcript must be a string")
    validated = [_validate_entity(entity) for entity in entities]
    previous_end = 0
    for entity in validated:
        if (
            entity.end > len(gold_transcript)
            or gold_transcript[entity.start : entity.end] != entity.text
        ):
            raise ValueError("gold entity span does not match gold_transcript")
        if entity.start < previous_end:
            raise ValueError("gold entities must be ordered and non-overlapping")
        previous_end = entity.end
    full_normalized = normalize_runtime(gold_transcript)
    projected: list[ProjectedEntity] = []
    normalized_position = 0
    for entity in validated:
        normalized_entity = normalize_runtime(entity.text)
        if not normalized_entity:
            raise ValueError("normalization removed a gold entity surface")
        prefix = normalize_runtime(gold_transcript[: entity.start])
        suffix = normalize_runtime(gold_transcript[entity.end :])
        candidate = full_normalized.find(normalized_entity, normalized_position)
        while candidate >= 0:
            candidate_end = candidate + len(normalized_entity)
            prefix_matches = full_normalized[:candidate].rstrip() == prefix
            suffix_matches = full_normalized[candidate_end:].lstrip() == suffix
            if prefix_matches and suffix_matches:
                projected.append(
                    ProjectedEntity(
                        text=normalized_entity,
                        label=entity.label,
                        start=candidate,
                        end=candidate_end,
                    )
                )
                normalized_position = candidate_end
                break
            candidate = full_normalized.find(normalized_entity, candidate + 1)
        else:
            raise ValueError("cannot preserve entity surfaces through runtime normalization")
    return tuple(projected)


compute_entity_metrics = calculate_entity_metrics
