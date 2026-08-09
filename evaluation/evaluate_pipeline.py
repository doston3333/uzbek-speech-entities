"""Run the required A-H normalization and NER ablation matrix."""

from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from time import monotonic
from typing import Any

from evaluation.benchmark_runtime import RuntimeMeasurement, measure_runtime
from evaluation.config import EvaluationConfig, load_evaluation_config
from evaluation.dataset import (
    APPLICATION_LABELS,
    EvaluationDataset,
    EvaluationSample,
    build_compliance_report,
    load_dataset,
    write_report,
)
from evaluation.entity_metrics import (
    EntityMetrics,
    EntityScore,
    align_entities_to_reference,
    calculate_entity_metrics,
    project_gold_entities,
)
from evaluation.io_utils import (
    read_jsonl,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from uzbek_speech_entities.config import AppConfig, load_config
from uzbek_speech_entities.ner.model_selection import load_run
from uzbek_speech_entities.ner.predictor import (
    NERPredictor,
    NERService,
)
from uzbek_speech_entities.normalization.runtime import (
    normalize_runtime,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Ablation:
    id: str
    source: str
    stt_model: str | None
    normalization: bool
    ner_run: str
    match_mode: str


ABLATIONS: tuple[Ablation, ...] = (
    Ablation("A", "audio", "base", False, "clean", "surface"),
    Ablation("B", "audio", "base", True, "clean", "surface"),
    Ablation("C", "audio", "base", True, "augmented", "surface"),
    Ablation("D", "audio", "small", False, "clean", "surface"),
    Ablation("E", "audio", "small", True, "clean", "surface"),
    Ablation("F", "audio", "small", True, "augmented", "surface"),
    Ablation("G", "gold", None, True, "clean", "span"),
    Ablation("H", "gold", None, True, "augmented", "span"),
)


@dataclass(frozen=True, slots=True)
class AblationEvaluation:
    summary: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]


def _build_ner_predictor(model_path: Any, app_config: AppConfig) -> NERPredictor:
    values = app_config.section("ner")
    max_length = values.get("max_length")
    threshold = values.get("confidence_threshold")
    labels = values.get("visible_labels")
    mapping = values.get("model_to_application_labels")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not isinstance(labels, tuple)
        or not all(isinstance(label, str) for label in labels)
        or not isinstance(mapping, Mapping)
        or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
        )
    ):
        raise ValueError("application NER configuration is invalid for evaluation")
    return NERPredictor(
        model_path,
        max_length=max_length,
        confidence_threshold=float(threshold),
        visible_labels=labels,
        model_to_application_labels=mapping,
        local_files_only=True,
    )


def _prediction_index(
    records: Sequence[Mapping[str, Any]],
    dataset: EvaluationDataset,
    expected_model_ids: Mapping[str, str],
) -> Mapping[tuple[str, str], Mapping[str, Any]]:
    expected_ids = {sample.id for sample in dataset.samples}
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    revisions: dict[str, str] = {}
    for row in records:
        model = row.get("model_key")
        sample_id = row.get("sample_id")
        transcript = row.get("raw_transcript")
        if model not in {"base", "small"} or not isinstance(sample_id, str):
            raise ValueError("STT prediction row has invalid model_key or sample_id")
        if sample_id not in expected_ids or not isinstance(transcript, str):
            raise ValueError("STT prediction row does not match the evaluation dataset")
        model_id = row.get("model_id")
        if model_id != expected_model_ids[model]:
            raise ValueError(f"STT prediction row has the wrong model_id for {model}")
        revision = row.get("resolved_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("STT prediction row must include a resolved_revision")
        previous_revision = revisions.setdefault(model, revision)
        if previous_revision != revision:
            raise ValueError(f"STT prediction rows mix revisions for {model}")
        key = (model, sample_id)
        if key in index:
            raise ValueError(f"duplicate STT prediction row: {model}/{sample_id}")
        index[key] = row
    expected = {(model, sample_id) for model in ("base", "small") for sample_id in expected_ids}
    missing = expected - set(index)
    if missing:
        formatted = ", ".join(f"{model}/{sample}" for model, sample in sorted(missing))
        raise ValueError(f"missing STT prediction rows: {formatted}")
    return index


def _entity_record(entity: object) -> dict[str, object]:
    text = getattr(entity, "text", None)
    label = getattr(entity, "label", None)
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    score = getattr(entity, "score", None)
    row: dict[str, object] = {"text": text, "label": label, "start": start, "end": end}
    if isinstance(score, int | float) and not isinstance(score, bool):
        row["score"] = float(score)
    return row


def _score_from_counts(tp: int, fp: int, fn: int, support: int) -> EntityScore:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return EntityScore(tp, fp, fn, support, precision, recall, f1)


def _aggregate_metrics(metrics: Sequence[EntityMetrics]) -> EntityMetrics:
    if not metrics:
        raise ValueError("at least one entity metric result is required")
    by_label: dict[str, EntityScore] = {}
    for label in APPLICATION_LABELS:
        values = [item.by_label[label] for item in metrics]
        by_label[label] = _score_from_counts(
            sum(item.true_positives for item in values),
            sum(item.false_positives for item in values),
            sum(item.false_negatives for item in values),
            sum(item.support for item in values),
        )
    overall_values = [item.overall for item in metrics]
    overall = _score_from_counts(
        sum(item.true_positives for item in overall_values),
        sum(item.false_positives for item in overall_values),
        sum(item.false_negatives for item in overall_values),
        sum(item.support for item in overall_values),
    )
    return EntityMetrics(
        by_label=by_label,
        overall=overall,
        macro_f1=fmean(by_label[label].f1 for label in APPLICATION_LABELS),
    )


def _source_text(
    ablation: Ablation,
    sample: EvaluationSample,
    stt_predictions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[str, str]:
    if ablation.source == "gold":
        return sample.gold_transcript, normalize_runtime(sample.gold_transcript)
    if ablation.stt_model is None:
        raise RuntimeError("audio ablation is missing its STT model")
    row = stt_predictions[(ablation.stt_model, sample.id)]
    raw = row.get("raw_transcript")
    if not isinstance(raw, str):
        raise ValueError("STT prediction transcript must be text")
    return raw, normalize_runtime(raw) if ablation.normalization else raw


def evaluate_ablation(
    *,
    ablation: Ablation,
    predictor: NERService,
    dataset: EvaluationDataset,
    stt_predictions: Mapping[tuple[str, str], Mapping[str, Any]],
    load_measurement: RuntimeMeasurement[None],
) -> AblationEvaluation:
    """Run one matrix row with explicit coordinate/matching semantics."""
    rows: list[Mapping[str, object]] = []
    sample_metrics: list[EntityMetrics] = []
    sample_exact_span_metrics: list[EntityMetrics] = []
    inference_seconds = 0.0
    for sample in dataset.samples:
        raw_transcript, ner_input = _source_text(ablation, sample, stt_predictions)
        started = monotonic()
        predicted = predictor.predict(ner_input)
        sample_inference_seconds = max(0.0, monotonic() - started)
        inference_seconds += sample_inference_seconds
        gold: Sequence[object]
        if ablation.match_mode == "span":
            gold = project_gold_entities(sample.gold_transcript, sample.entities)
            metrics = calculate_entity_metrics(gold, predicted, mode="span")
            exact_span_metrics = metrics
            span_alignment = "native-normalized-offsets"
        else:
            gold = sample.entities
            metrics = calculate_entity_metrics(gold, predicted, mode="surface")
            normalized_gold = normalize_runtime(sample.gold_transcript)
            projected_gold = project_gold_entities(sample.gold_transcript, sample.entities)
            aligned_predicted = align_entities_to_reference(normalized_gold, ner_input, predicted)
            exact_span_metrics = calculate_entity_metrics(
                projected_gold, aligned_predicted, mode="span"
            )
            span_alignment = "exact-token-alignment"
        sample_metrics.append(metrics)
        sample_exact_span_metrics.append(exact_span_metrics)
        rows.append(
            {
                "schema_version": 1,
                "ablation": ablation.id,
                "sample_id": sample.id,
                "source": ablation.source,
                "stt_model": ablation.stt_model,
                "normalization": ablation.normalization,
                "ner_run": ablation.ner_run,
                "match_mode": ablation.match_mode,
                "span_alignment": span_alignment,
                "gold_transcript": sample.gold_transcript,
                "raw_transcript": raw_transcript,
                "normalized_transcript": ner_input,
                "gold_entities": [_entity_record(entity) for entity in gold],
                "predicted_entities": [_entity_record(entity) for entity in predicted],
                "metrics": {
                    "by_label": {
                        label: asdict(metrics.by_label[label]) for label in APPLICATION_LABELS
                    },
                    "overall": asdict(metrics.overall),
                    "four_class_macro_f1": metrics.macro_f1,
                    "overall_exact_span": asdict(exact_span_metrics.overall),
                },
                "ner_inference_seconds": sample_inference_seconds,
                "conditions": list(sample.conditions),
            }
        )
    aggregate = _aggregate_metrics(sample_metrics)
    exact_span_aggregate = _aggregate_metrics(sample_exact_span_metrics)
    summary: dict[str, object] = {
        "ablation": ablation.id,
        "source": ablation.source,
        "stt_model": ablation.stt_model,
        "normalization": ablation.normalization,
        "ner_run": ablation.ner_run,
        "match_mode": ablation.match_mode,
        "sample_count": len(dataset.samples),
        "overall_precision": aggregate.overall.precision,
        "overall_recall": aggregate.overall.recall,
        "overall_exact_entity_f1": aggregate.overall.f1,
        "overall_exact_span_f1": exact_span_aggregate.overall.f1,
        "four_class_macro_f1": aggregate.macro_f1,
        "ner_model_loading_seconds": load_measurement.elapsed_seconds,
        "ner_model_load_peak_rss_mb": load_measurement.peak_rss_mb,
        "ner_inference_seconds": inference_seconds,
        "mean_ner_inference_seconds": inference_seconds / len(dataset.samples),
    }
    for label in APPLICATION_LABELS:
        score = aggregate.by_label[label]
        summary[f"{label}_precision"] = score.precision
        summary[f"{label}_recall"] = score.recall
        summary[f"{label}_f1"] = score.f1
        summary[f"{label}_count"] = score.support
    return AblationEvaluation(summary=summary, predictions=tuple(rows))


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        return


def _selection_report(
    summaries: Sequence[Mapping[str, object]], *, dataset_compliant: bool
) -> Mapping[str, object]:
    by_id = {str(row["ablation"]): row for row in summaries}

    def number(row: Mapping[str, object], key: str) -> float:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise RuntimeError(f"pipeline summary field is not numeric: {key}")
        return float(value)

    def run_values(ids: tuple[str, str]) -> tuple[float, float, float, float]:
        rows = [by_id[item] for item in ids]
        macro = fmean(number(row, "four_class_macro_f1") for row in rows)
        weakest = min(
            fmean(number(row, f"{label}_f1") for row in rows) for label in APPLICATION_LABELS
        )
        overall = fmean(number(row, "overall_exact_entity_f1") for row in rows)
        latency = fmean(number(row, "mean_ner_inference_seconds") for row in rows)
        return macro, weakest, overall, latency

    clean = run_values(("B", "E"))
    augmented = run_values(("C", "F"))
    clean_rank = (clean[0], clean[1], clean[2], -clean[3])
    augmented_rank = (augmented[0], augmented[1], augmented[2], -augmented[3])
    selected = "augmented" if augmented_rank > clean_rank else "clean"
    return {
        "selected_run": selected,
        "provisional": not dataset_compliant,
        "promotion_performed": False,
        "criteria": [
            "higher normalized-audio four-class macro F1 across Base and Small",
            "higher weakest public-class F1",
            "higher overall exact-entity F1",
            "lower NER inference latency",
            "exact ties prefer clean conservatively",
        ],
        "clean": {
            "four_class_macro_f1": clean[0],
            "weakest_class_f1": clean[1],
            "overall_exact_entity_f1": clean[2],
            "mean_inference_seconds": clean[3],
        },
        "augmented": {
            "four_class_macro_f1": augmented[0],
            "weakest_class_f1": augmented[1],
            "overall_exact_entity_f1": augmented[2],
            "mean_inference_seconds": augmented[3],
        },
        "limitation": (
            "A model is never promoted automatically. A compliant private dataset and manual "
            "review of per-class results are required."
        ),
    }


def run_evaluation(
    config: EvaluationConfig, *, allow_incomplete_dataset: bool
) -> tuple[Mapping[str, object], ...]:
    dataset = load_dataset(config.metadata_path, require_files=True)
    compliance = build_compliance_report(dataset, inspect_files=True)
    write_report(compliance, config.compliance_report_path)
    if config.require_compliance and not compliance.compliant and not allow_incomplete_dataset:
        issues = "\n- ".join(compliance.issues)
        raise ValueError(f"evaluation dataset is not Phase 8 compliant:\n- {issues}")
    records = read_jsonl(config.stt_predictions_path)
    prediction_index = _prediction_index(records, dataset, config.stt_model_ids)
    app_config = load_config()
    run_paths = {
        "clean": config.clean_ner_run_path,
        "augmented": config.augmented_ner_run_path,
    }
    summaries: list[Mapping[str, object]] = []
    predictions: list[Mapping[str, object]] = []
    for run_name in ("clean", "augmented"):
        run = load_run(run_name, run_paths[run_name])
        predictor = _build_ner_predictor(run.checkpoint, app_config)
        load_measurement = measure_runtime(predictor.load)
        for ablation in ABLATIONS:
            if ablation.ner_run != run_name:
                continue
            result = evaluate_ablation(
                ablation=ablation,
                predictor=predictor,
                dataset=dataset,
                stt_predictions=prediction_index,
                load_measurement=load_measurement,
            )
            summaries.append(result.summary)
            predictions.extend(result.predictions)
        del predictor
        _release_accelerator_memory()
    summaries.sort(key=lambda row: str(row["ablation"]))
    predictions.sort(key=lambda row: (str(row["ablation"]), str(row["sample_id"])))
    fieldnames = [
        "ablation",
        "source",
        "stt_model",
        "normalization",
        "ner_run",
        "match_mode",
        "sample_count",
        "PER_precision",
        "PER_recall",
        "PER_f1",
        "PER_count",
        "LOC_precision",
        "LOC_recall",
        "LOC_f1",
        "LOC_count",
        "ORG_precision",
        "ORG_recall",
        "ORG_f1",
        "ORG_count",
        "DATE_precision",
        "DATE_recall",
        "DATE_f1",
        "DATE_count",
        "four_class_macro_f1",
        "overall_precision",
        "overall_recall",
        "overall_exact_entity_f1",
        "overall_exact_span_f1",
        "ner_model_loading_seconds",
        "ner_model_load_peak_rss_mb",
        "ner_inference_seconds",
        "mean_ner_inference_seconds",
    ]
    write_csv_atomic(summaries, fieldnames, config.pipeline_summary_path)
    write_jsonl_atomic(predictions, config.pipeline_predictions_path)
    write_json_atomic(
        _selection_report(summaries, dataset_compliant=compliance.compliant),
        config.pipeline_selection_path,
    )
    return tuple(summaries)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--allow-incomplete-dataset", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_args(argv)
    try:
        run_evaluation(
            load_evaluation_config(arguments.config),
            allow_incomplete_dataset=bool(arguments.allow_incomplete_dataset),
        )
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("pipeline evaluation failed: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
