"""Transcript and entity-mention metrics for private evaluation data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast

import jiwer

from uzbek_speech_entities.normalization import normalize_evaluation

APPLICATION_LABELS: Final[tuple[str, ...]] = ("PER", "LOC", "ORG", "DATE")


@dataclass(frozen=True, slots=True)
class ErrorRate:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    rate: float | None


@dataclass(frozen=True, slots=True)
class TranscriptMetrics:
    raw_wer: ErrorRate
    normalized_wer: ErrorRate
    normalized_cer: ErrorRate


@dataclass(frozen=True, slots=True)
class MentionAccuracy:
    correct: int
    total: int
    accuracy: float | None


class _JiwerOutput(Protocol):
    substitutions: int
    deletions: int
    insertions: int


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _error_rate(output: _JiwerOutput, denominator: int) -> ErrorRate:
    substitutions = int(output.substitutions)
    deletions = int(output.deletions)
    insertions = int(output.insertions)
    errors = substitutions + deletions + insertions
    return ErrorRate(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_units=denominator,
        rate=errors / denominator if denominator else None,
    )


def _word_rate(reference: str, hypothesis: str) -> ErrorRate:
    result = jiwer.process_words(reference, hypothesis)
    return _error_rate(cast(_JiwerOutput, result), len(reference.split()))


def _character_rate(reference: str, hypothesis: str) -> ErrorRate:
    result = jiwer.process_characters(reference, hypothesis)
    return _error_rate(cast(_JiwerOutput, result), len(reference))


def calculate_transcript_metrics(reference: str, hypothesis: str) -> TranscriptMetrics:
    """Calculate raw WER plus evaluation-normalized WER and CER for one sample."""
    reference = _require_text(reference, "reference")
    hypothesis = _require_text(hypothesis, "hypothesis")
    normalized_reference = normalize_evaluation(reference)
    normalized_hypothesis = normalize_evaluation(hypothesis)
    return TranscriptMetrics(
        raw_wer=_word_rate(reference, hypothesis),
        normalized_wer=_word_rate(normalized_reference, normalized_hypothesis),
        normalized_cer=_character_rate(normalized_reference, normalized_hypothesis),
    )


def aggregate_transcript_metrics(
    samples: Iterable[tuple[str, str]],
) -> TranscriptMetrics:
    """Aggregate edit counts, yielding corpus-weighted rather than mean sample rates."""
    values = list(samples)
    if not values:
        raise ValueError("at least one transcript sample is required")
    metrics = [
        calculate_transcript_metrics(reference, hypothesis) for reference, hypothesis in values
    ]

    def aggregate(rates: Iterable[ErrorRate]) -> ErrorRate:
        items = list(rates)
        substitutions = sum(item.substitutions for item in items)
        deletions = sum(item.deletions for item in items)
        insertions = sum(item.insertions for item in items)
        reference_units = sum(item.reference_units for item in items)
        errors = substitutions + deletions + insertions
        return ErrorRate(
            substitutions=substitutions,
            deletions=deletions,
            insertions=insertions,
            reference_units=reference_units,
            rate=errors / reference_units if reference_units else None,
        )

    return TranscriptMetrics(
        raw_wer=aggregate(item.raw_wer for item in metrics),
        normalized_wer=aggregate(item.normalized_wer for item in metrics),
        normalized_cer=aggregate(item.normalized_cer for item in metrics),
    )


def _occurrence_count(tokens: Sequence[str], phrase: Sequence[str]) -> int:
    if not phrase:
        return 0
    width = len(phrase)
    return sum(
        tokens[index : index + width] == list(phrase)
        for index in range(len(tokens) - width + 1)
    )


def entity_mention_accuracy(
    gold_entities: Iterable[object], hypothesis: str
) -> Mapping[str, MentionAccuracy]:
    """Score exact mention occurrences per public label, capped by gold multiplicity."""
    hypothesis = _require_text(hypothesis, "hypothesis")
    gold = list(gold_entities)
    normalized_hypothesis = normalize_evaluation(hypothesis).split()
    result: dict[str, MentionAccuracy] = {}
    for label in APPLICATION_LABELS:
        mentions: Counter[tuple[str, ...]] = Counter()
        for entity in gold:
            entity_label = getattr(entity, "label", None)
            text = getattr(entity, "text", None)
            if entity_label not in APPLICATION_LABELS or not isinstance(text, str):
                raise ValueError("gold entities must have a public label and string text")
            if entity_label == label:
                normalized = tuple(normalize_evaluation(text).split())
                if not normalized:
                    raise ValueError("gold entity text must have a non-empty normalized surface")
                mentions[normalized] += 1
        total = sum(mentions.values())
        correct = sum(
            min(count, _occurrence_count(normalized_hypothesis, phrase))
            for phrase, count in mentions.items()
        )
        result[label] = MentionAccuracy(
            correct=correct, total=total, accuracy=correct / total if total else None
        )
    return result


# Short aliases keep downstream evaluation scripts readable.
compute_transcript_metrics = calculate_transcript_metrics
corpus_transcript_metrics = aggregate_transcript_metrics
compute_entity_mention_accuracy = entity_mention_accuracy
