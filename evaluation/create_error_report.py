"""Create privacy-safe summaries and local detailed Phase 8 error records."""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jiwer

from evaluation.config import EvaluationConfig, load_evaluation_config
from evaluation.dataset import (
    APPLICATION_LABELS,
    build_compliance_report,
    load_dataset,
    write_report,
)
from evaluation.io_utils import read_jsonl, write_jsonl_atomic, write_text_atomic
from uzbek_speech_entities.normalization.evaluation import (
    normalize_evaluation,
)
from uzbek_speech_entities.normalization.runtime import (
    normalize_runtime,
)

LOGGER = logging.getLogger(__name__)

STT_ERROR_CATEGORIES = (
    "Person mistranscribed",
    "Location mistranscribed",
    "Organization mistranscribed",
    "Date mistranscribed",
    "Time mistranscribed",
    "Word omitted",
    "Word inserted",
    "Repetition",
    "Apostrophe error",
    "Numeric-format error",
    "Suffix error",
    "Code-switching error",
    "Noise-related error",
)
NER_ERROR_CATEGORIES = (
    "Missed person",
    "Missed location",
    "Missed organization",
    "Missed date",
    "Person classified as location",
    "Person classified as organization",
    "Location classified as organization",
    "Organization classified as location",
    "Date classified as number only",
    "Number incorrectly classified as date",
    "Incomplete entity",
    "Excessive entity span",
    "False positive",
    "Low-confidence correct prediction",
)
NORMALIZATION_ERROR_CATEGORIES = (
    "Incorrect apostrophe replacement",
    "Incorrect date punctuation",
    "Incorrect time punctuation",
    "Whitespace corruption",
    "Meaning changed",
    "Offset mismatch",
    "Unsafe autocorrection",
    "Unsafe date resolution",
)
_APOSTROPHES = "'’‘ʻʼ`"
_DIGIT_RE = re.compile(r"\d+")
_PUBLIC_LABEL_TO_STT = {
    "PER": "Person mistranscribed",
    "LOC": "Location mistranscribed",
    "ORG": "Organization mistranscribed",
    "DATE": "Date mistranscribed",
}
_PUBLIC_LABEL_TO_MISSED = {
    "PER": "Missed person",
    "LOC": "Missed location",
    "ORG": "Missed organization",
    "DATE": "Missed date",
}
_CONFUSIONS = {
    ("PER", "LOC"): "Person classified as location",
    ("PER", "ORG"): "Person classified as organization",
    ("LOC", "ORG"): "Location classified as organization",
    ("ORG", "LOC"): "Organization classified as location",
}


def _text(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"error-analysis input {key} must be text")
    return value


def _entities(row: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"error-analysis input {key} must be an entity array")
    entities = list(value)
    for entity in entities:
        if entity.get("label") not in APPLICATION_LABELS or not isinstance(
            entity.get("text"), str
        ):
            raise ValueError(f"error-analysis input {key} contains an invalid entity")
    return entities


def _surface(entity: Mapping[str, Any]) -> str:
    return str(normalize_evaluation(str(entity["text"])))


def _category_list(row: Mapping[str, object], key: str) -> list[str]:
    value = row.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"error record {key} must be a text array")
    return value


def _phrase_occurs(phrase: str, transcript: str) -> bool:
    phrase_tokens = phrase.split()
    transcript_tokens = transcript.split()
    width = len(phrase_tokens)
    return bool(width) and any(
        transcript_tokens[index : index + width] == phrase_tokens
        for index in range(len(transcript_tokens) - width + 1)
    )


def classify_stt_errors(
    gold_transcript: str,
    raw_transcript: str,
    gold_entities: Iterable[Mapping[str, Any]],
    conditions: Iterable[str],
) -> tuple[str, ...]:
    reference = normalize_evaluation(gold_transcript)
    hypothesis = normalize_evaluation(raw_transcript)
    if reference == hypothesis:
        return ()
    errors: set[str] = set()
    for entity in gold_entities:
        phrase = _surface(entity)
        if not _phrase_occurs(phrase, hypothesis):
            label = str(entity["label"])
            if label == "DATE" and ":" in str(entity["text"]):
                errors.add("Time mistranscribed")
            else:
                errors.add(_PUBLIC_LABEL_TO_STT[label])
    alignment = jiwer.process_words(reference, hypothesis)
    if alignment.deletions:
        errors.add("Word omitted")
    if alignment.insertions:
        errors.add("Word inserted")
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    if any(
        count > Counter(reference_tokens)[token]
        for token, count in Counter(hypothesis_tokens).items()
        if count > 1
    ):
        errors.add("Repetition")
    remove_apostrophes = str.maketrans("", "", _APOSTROPHES)
    if reference.translate(remove_apostrophes) == hypothesis.translate(remove_apostrophes):
        errors.add("Apostrophe error")
    if _DIGIT_RE.findall(reference) != _DIGIT_RE.findall(hypothesis) and (
        _DIGIT_RE.search(reference) or _DIGIT_RE.search(hypothesis)
    ):
        errors.add("Numeric-format error")
    normalized_conditions = set(conditions)
    if "uzbek_russian_code_switching" in normalized_conditions:
        errors.add("Code-switching error")
    if "background_noise" in normalized_conditions:
        errors.add("Noise-related error")
    return tuple(category for category in STT_ERROR_CATEGORIES if category in errors)


def classify_ner_errors(
    gold_entities: Sequence[Mapping[str, Any]],
    predicted_entities: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: set[str] = set()
    matched_predictions: set[int] = set()
    for gold in gold_entities:
        gold_label = str(gold["label"])
        gold_surface = _surface(gold)
        exact_index = next(
            (
                index
                for index, predicted in enumerate(predicted_entities)
                if index not in matched_predictions
                and predicted.get("label") == gold_label
                and _surface(predicted) == gold_surface
            ),
            None,
        )
        if exact_index is not None:
            matched_predictions.add(exact_index)
            score = predicted_entities[exact_index].get("score")
            if isinstance(score, int | float) and not isinstance(score, bool) and score < 0.6:
                errors.add("Low-confidence correct prediction")
            continue
        confusion_index = next(
            (
                index
                for index, predicted in enumerate(predicted_entities)
                if index not in matched_predictions and _surface(predicted) == gold_surface
            ),
            None,
        )
        if confusion_index is not None:
            predicted_label = str(predicted_entities[confusion_index]["label"])
            matched_predictions.add(confusion_index)
            confusion = _CONFUSIONS.get((gold_label, predicted_label))
            if confusion:
                errors.add(confusion)
            else:
                errors.add(_PUBLIC_LABEL_TO_MISSED[gold_label])
            continue
        partial_index = next(
            (
                index
                for index, predicted in enumerate(predicted_entities)
                if index not in matched_predictions
                and predicted.get("label") == gold_label
                and (
                    _surface(predicted) in gold_surface or gold_surface in _surface(predicted)
                )
            ),
            None,
        )
        if partial_index is not None:
            predicted_surface = _surface(predicted_entities[partial_index])
            matched_predictions.add(partial_index)
            errors.add(
                "Incomplete entity"
                if len(predicted_surface) < len(gold_surface)
                else "Excessive entity span"
            )
            continue
        errors.add(_PUBLIC_LABEL_TO_MISSED[gold_label])
        if gold_label == "DATE" and _DIGIT_RE.search(gold_surface):
            errors.add("Date classified as number only")
    for index, predicted in enumerate(predicted_entities):
        if index in matched_predictions:
            continue
        errors.add("False positive")
        if predicted.get("label") == "DATE" and _DIGIT_RE.fullmatch(_surface(predicted)):
            errors.add("Number incorrectly classified as date")
    return tuple(category for category in NER_ERROR_CATEGORIES if category in errors)


def classify_normalization_errors(
    raw_transcript: str,
    normalized_transcript: str,
    predicted_entities: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    errors: set[str] = set()
    expected = normalize_runtime(raw_transcript)
    if normalized_transcript != expected:
        if "  " in normalized_transcript or normalized_transcript != normalized_transcript.strip():
            errors.add("Whitespace corruption")
        if normalize_evaluation(raw_transcript) != normalize_evaluation(normalized_transcript):
            errors.add("Meaning changed")
        if any(character in normalized_transcript for character in _APOSTROPHES.replace("ʻ", "")):
            errors.add("Incorrect apostrophe replacement")
        if _DIGIT_RE.search(raw_transcript) and " - " in normalized_transcript:
            errors.add("Incorrect date punctuation")
        if re.search(r"\d\s+:\s+\d", normalized_transcript):
            errors.add("Incorrect time punctuation")
    for entity in predicted_entities:
        start, end, text = entity.get("start"), entity.get("end"), entity.get("text")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not isinstance(text, str)
            or start < 0
            or end <= start
            or end > len(normalized_transcript)
            or normalized_transcript[start:end] != text
        ):
            errors.add("Offset mismatch")
    return tuple(
        category for category in NORMALIZATION_ERROR_CATEGORIES if category in errors
    )


def build_error_records(
    pipeline_rows: Sequence[Mapping[str, Any]], *, primary_ablation: str
) -> tuple[Mapping[str, object], ...]:
    selected = [row for row in pipeline_rows if row.get("ablation") == primary_ablation]
    if not selected:
        raise ValueError(f"no pipeline predictions found for ablation {primary_ablation}")
    records: list[Mapping[str, object]] = []
    for row in selected:
        gold = _entities(row, "gold_entities")
        predicted = _entities(row, "predicted_entities")
        conditions_raw = row.get("conditions")
        if not isinstance(conditions_raw, list) or not all(
            isinstance(item, str) for item in conditions_raw
        ):
            raise ValueError("error-analysis input conditions must be a text array")
        conditions = list(conditions_raw)
        gold_transcript = _text(row, "gold_transcript")
        raw_transcript = _text(row, "raw_transcript")
        normalized_transcript = _text(row, "normalized_transcript")
        records.append(
            {
                "audio_id": _text(row, "sample_id"),
                "gold_transcript": gold_transcript,
                "raw_transcript": raw_transcript,
                "normalized_transcript": normalized_transcript,
                "gold_entities": gold,
                "predicted_entities": predicted,
                "stt_error": list(
                    classify_stt_errors(
                        gold_transcript, raw_transcript, gold, conditions
                    )
                ),
                "ner_error": list(classify_ner_errors(gold, predicted)),
                "normalization_error": list(
                    classify_normalization_errors(
                        raw_transcript, normalized_transcript, predicted
                    )
                ),
                "notes": f"conditions: {', '.join(conditions)}; automatic taxonomy needs review",
            }
        )
    return tuple(records)


def _markdown_summary(
    records: Sequence[Mapping[str, object]],
    config: EvaluationConfig,
    *,
    dataset_compliant: bool,
) -> str:
    stt_counts: Counter[str] = Counter()
    ner_counts: Counter[str] = Counter()
    normalization_counts: Counter[str] = Counter()
    for row in records:
        stt_counts.update(_category_list(row, "stt_error"))
        ner_counts.update(_category_list(row, "ner_error"))
        normalization_counts.update(_category_list(row, "normalization_error"))

    def table(title: str, categories: Sequence[str], counts: Counter[str]) -> list[str]:
        lines = [f"## {title}", "", "| Category | Count |", "| --- | ---: |"]
        lines.extend(f"| {category} | {counts[category]} |" for category in categories)
        lines.append("")
        return lines

    lines = [
        "# Error analysis",
        "",
        f"Primary ablation: **{config.primary_ablation}**",
        f"Evaluated recordings: **{len(records)}**",
        f"Dataset status: **{'Phase 8 compliant' if dataset_compliant else 'PROVISIONAL'}**",
        "",
        (
            "This aggregate summary contains no transcripts. Full required error records are "
            f"local, Git-ignored, and stored at `{config.error_records_path}`."
        ),
        "",
        "Automatic classifications are an auditable first pass and require manual review "
        "of `notes`.",
        "",
    ]
    lines.extend(table("STT errors", STT_ERROR_CATEGORIES, stt_counts))
    lines.extend(table("NER errors", NER_ERROR_CATEGORIES, ner_counts))
    lines.extend(
        table("Normalization errors", NORMALIZATION_ERROR_CATEGORIES, normalization_counts)
    )
    return "\n".join(lines).rstrip() + "\n"


def run_report(
    config: EvaluationConfig, *, allow_incomplete_dataset: bool = False
) -> tuple[Mapping[str, object], ...]:
    dataset = load_dataset(config.metadata_path, require_files=True)
    compliance = build_compliance_report(dataset, inspect_files=True)
    write_report(compliance, config.compliance_report_path)
    if config.require_compliance and not compliance.compliant and not allow_incomplete_dataset:
        issues = "\n- ".join(compliance.issues)
        raise ValueError(f"evaluation dataset is not Phase 8 compliant:\n- {issues}")
    rows = read_jsonl(config.pipeline_predictions_path)
    records = build_error_records(rows, primary_ablation=config.primary_ablation)
    record_ids = [str(record["audio_id"]) for record in records]
    expected_ids = {sample.id for sample in dataset.samples}
    if len(record_ids) != len(set(record_ids)) or set(record_ids) != expected_ids:
        raise ValueError("primary-ablation predictions do not exactly cover the evaluation dataset")
    write_jsonl_atomic(records, config.error_records_path)
    write_text_atomic(
        _markdown_summary(records, config, dataset_compliant=compliance.compliant),
        config.error_summary_path,
    )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_args(argv)
    try:
        run_report(
            load_evaluation_config(arguments.config),
            allow_incomplete_dataset=bool(arguments.allow_incomplete_dataset),
        )
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("error report failed: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
