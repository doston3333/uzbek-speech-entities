"""Compare local Uzbek Whisper Base and Small on a validated private corpus."""

from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic

import soundfile as sf  # type: ignore[import-untyped]
from huggingface_hub import snapshot_download

from evaluation.benchmark_runtime import measure_runtime
from evaluation.config import EvaluationConfig, load_evaluation_config
from evaluation.dataset import (
    APPLICATION_LABELS,
    EvaluationDataset,
    GoldEntity,
    build_compliance_report,
    load_dataset,
    write_report,
)
from evaluation.io_utils import write_csv_atomic, write_jsonl_atomic
from evaluation.transcript_metrics import (
    ErrorRate,
    aggregate_transcript_metrics,
    calculate_transcript_metrics,
    entity_mention_accuracy,
)
from uzbek_speech_entities.audio.preprocessing import prepared_audio
from uzbek_speech_entities.audio.validation import (
    AudioValidationConfig,
)
from uzbek_speech_entities.config import (
    AppConfig,
    load_config,
    resolve_project_path,
)
from uzbek_speech_entities.normalization.runtime import (
    normalize_runtime,
)
from uzbek_speech_entities.stt.base import SpeechToTextService, validate_immutable_revision
from uzbek_speech_entities.stt.transformers_backend import (
    TransformersSpeechToTextService,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class STTModelEvaluation:
    summary: Mapping[str, object]
    predictions: tuple[Mapping[str, object], ...]


def _entity_record(entity: GoldEntity) -> dict[str, object]:
    return {"text": entity.text, "label": entity.label, "start": entity.start, "end": entity.end}


def _rate_record(rate: ErrorRate) -> dict[str, object]:
    return asdict(rate)


def _audio_duration(path: Path) -> float:
    info = sf.info(path)
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError(f"audio duration is invalid: {path}")
    return float(info.frames / info.samplerate)


def _build_stt_service(
    model_id: str, revision: str, app_config: AppConfig, *, local_files_only: bool
) -> TransformersSpeechToTextService:
    values = app_config.section("stt")
    cache_dir = resolve_project_path("./models/cache")
    language = values.get("language")
    task = values.get("task")
    chunk_seconds = values.get("chunk_length_seconds")
    batch_size = values.get("batch_size")
    preference = values.get("device_preference")
    if (
        not isinstance(language, str)
        or not isinstance(task, str)
        or isinstance(chunk_seconds, bool)
        or not isinstance(chunk_seconds, int | float)
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not isinstance(preference, tuple)
        or not all(isinstance(item, str) for item in preference)
    ):
        raise ValueError("application STT configuration is invalid for evaluation")
    return TransformersSpeechToTextService(
        model_id=model_id,
        cache_dir=cache_dir,
        revision=validate_immutable_revision(revision, field_name="STT evaluation revision"),
        language=language,
        task=task,
        chunk_length_seconds=float(chunk_seconds),
        batch_size=batch_size,
        device_preference=preference,
        local_files_only=local_files_only,
    )


def _resolved_revision(model_id: str, revision: str, *, local_files_only: bool) -> str:
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=validate_immutable_revision(revision, field_name="STT evaluation revision"),
        cache_dir=str(resolve_project_path("./models/cache")),
        local_files_only=local_files_only,
    )
    return validate_immutable_revision(
        Path(snapshot).name, field_name="resolved STT evaluation revision"
    )


def _configured_model_revision(model_key: str, model_id: str, app_config: AppConfig) -> str:
    """Return the pinned app revision for an evaluation model, never a mutable ref."""
    if model_key == "small":
        id_key, revision_key = "model_id", "model_revision"
    elif model_key == "base":
        id_key, revision_key = "fallback_model_id", "fallback_model_revision"
    else:
        raise ValueError("model_key must be base or small")
    stt = app_config.section("stt")
    configured_id = stt.get(id_key)
    if not isinstance(configured_id, str) or configured_id.strip() != model_id:
        raise ValueError(f"evaluation model ID for {model_key} must match app STT configuration")
    return validate_immutable_revision(stt.get(revision_key), field_name=f"stt.{revision_key}")


def evaluate_stt_service(
    *,
    model_key: str,
    service: SpeechToTextService,
    dataset: EvaluationDataset,
    audio_config: AudioValidationConfig,
    resolved_revision: str,
) -> STTModelEvaluation:
    """Evaluate one already-constructed service without loading any other model."""
    if model_key not in {"base", "small"}:
        raise ValueError("model_key must be base or small")
    load_measurement = measure_runtime(service.load)

    def transcribe_all() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for sample in dataset.samples:
            duration_seconds = _audio_duration(sample.file)
            processing_started = monotonic()
            preprocessing_started = monotonic()
            with prepared_audio(sample.file, audio_config) as canonical_audio:
                preprocessing_seconds = max(0.0, monotonic() - preprocessing_started)
                stt_started = monotonic()
                raw_transcript = service.transcribe(canonical_audio)
                stt_seconds = max(0.0, monotonic() - stt_started)
            processing_seconds = max(0.0, monotonic() - processing_started)
            transcript_metrics = calculate_transcript_metrics(
                sample.gold_transcript, raw_transcript
            )
            mentions = entity_mention_accuracy(sample.entities, raw_transcript)
            rows.append(
                {
                    "schema_version": 1,
                    "sample_id": sample.id,
                    "model_key": model_key,
                    "model_id": service.model_id,
                    "resolved_revision": resolved_revision,
                    "device": service.device,
                    "audio_file": str(sample.file),
                    "audio_duration_seconds": duration_seconds,
                    "gold_transcript": sample.gold_transcript,
                    "raw_transcript": raw_transcript,
                    "normalized_transcript": normalize_runtime(raw_transcript),
                    "gold_entities": [_entity_record(entity) for entity in sample.entities],
                    "speaker_id": sample.speaker_id,
                    "conditions": list(sample.conditions),
                    "metrics": {
                        "raw_wer": _rate_record(transcript_metrics.raw_wer),
                        "normalized_wer": _rate_record(transcript_metrics.normalized_wer),
                        "normalized_cer": _rate_record(transcript_metrics.normalized_cer),
                        "mention_accuracy": {
                            label: asdict(mentions[label]) for label in APPLICATION_LABELS
                        },
                    },
                    "timing": {
                        "audio_preprocessing_seconds": preprocessing_seconds,
                        "stt_seconds": stt_seconds,
                        "audio_processing_seconds": processing_seconds,
                        "real_time_factor": processing_seconds / duration_seconds,
                    },
                }
            )
        return rows

    inference_measurement = measure_runtime(transcribe_all)
    predictions = inference_measurement.value
    transcript_pairs = [
        (str(row["gold_transcript"]), str(row["raw_transcript"])) for row in predictions
    ]
    aggregate = aggregate_transcript_metrics(transcript_pairs)
    mention_totals: dict[str, tuple[int, int]] = {}
    for label in APPLICATION_LABELS:
        correct = 0
        total = 0
        for row in predictions:
            metrics = row["metrics"]
            if not isinstance(metrics, Mapping):
                raise RuntimeError("internal STT metric row is invalid")
            mention_values = metrics["mention_accuracy"]
            if not isinstance(mention_values, Mapping):
                raise RuntimeError("internal STT mention metric row is invalid")
            label_values = mention_values[label]
            if not isinstance(label_values, Mapping):
                raise RuntimeError("internal STT label metric row is invalid")
            correct += int(label_values["correct"])
            total += int(label_values["total"])
        mention_totals[label] = (correct, total)
    total_audio_seconds = sum(_numeric_value(row, "audio_duration_seconds") for row in predictions)
    total_processing_seconds = sum(
        _timing_value(row, "audio_processing_seconds") for row in predictions
    )
    summary: dict[str, object] = {
        "model_key": model_key,
        "model_id": service.model_id,
        "resolved_revision": resolved_revision,
        "device": service.device,
        "sample_count": len(predictions),
        "total_audio_seconds": total_audio_seconds,
        "raw_wer": aggregate.raw_wer.rate,
        "normalized_wer": aggregate.normalized_wer.rate,
        "cer": aggregate.normalized_cer.rate,
        "model_loading_seconds": load_measurement.elapsed_seconds,
        "audio_processing_seconds": total_processing_seconds,
        "mean_audio_processing_seconds": total_processing_seconds / len(predictions),
        "real_time_factor": total_processing_seconds / total_audio_seconds,
        "load_peak_rss_mb": load_measurement.peak_rss_mb,
        "load_peak_rss_delta_mb": load_measurement.peak_rss_delta_mb,
        "inference_peak_rss_mb": inference_measurement.peak_rss_mb,
        "inference_peak_rss_delta_mb": inference_measurement.peak_rss_delta_mb,
    }
    for label in APPLICATION_LABELS:
        correct, total = mention_totals[label]
        summary[f"{label}_mention_correct"] = correct
        summary[f"{label}_mention_total"] = total
        summary[f"{label}_mention_accuracy"] = correct / total if total else None
    return STTModelEvaluation(summary=summary, predictions=tuple(predictions))


def _timing_value(row: Mapping[str, object], key: str) -> float:
    timing = row.get("timing")
    if not isinstance(timing, Mapping):
        raise RuntimeError("internal STT timing row is invalid")
    value = timing.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError("internal STT timing value is invalid")
    return float(value)


def _numeric_value(row: Mapping[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"internal numeric field is invalid: {key}")
    return float(value)


def _release_accelerator_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, RuntimeError):
        return


def run_evaluation(
    config: EvaluationConfig,
    *,
    selected_models: Sequence[str],
    allow_incomplete_dataset: bool,
    allow_download: bool,
) -> tuple[Mapping[str, object], ...]:
    dataset = load_dataset(config.metadata_path, require_files=True)
    compliance = build_compliance_report(dataset, inspect_files=True)
    write_report(compliance, config.compliance_report_path)
    if config.require_compliance and not compliance.compliant and not allow_incomplete_dataset:
        issues = "\n- ".join(compliance.issues)
        raise ValueError(f"evaluation dataset is not Phase 8 compliant:\n- {issues}")
    app_config = load_config()
    audio_config = AudioValidationConfig.from_mapping(app_config.section("audio"))
    summaries: list[Mapping[str, object]] = []
    predictions: list[Mapping[str, object]] = []
    local_files_only = config.stt_local_files_only and not allow_download
    for model_key in selected_models:
        model_id = config.stt_model_ids[model_key]
        revision = _configured_model_revision(model_key, model_id, app_config)
        LOGGER.info("Evaluating %s (%s@%s)", model_key, model_id, revision)
        service = _build_stt_service(
            model_id, revision, app_config, local_files_only=local_files_only
        )
        result = evaluate_stt_service(
            model_key=model_key,
            service=service,
            dataset=dataset,
            audio_config=audio_config,
            resolved_revision=_resolved_revision(
                model_id, revision, local_files_only=local_files_only
            ),
        )
        summaries.append(result.summary)
        predictions.extend(result.predictions)
        del service
        _release_accelerator_memory()
    fieldnames = [
        "model_key",
        "model_id",
        "resolved_revision",
        "device",
        "sample_count",
        "total_audio_seconds",
        "raw_wer",
        "normalized_wer",
        "cer",
        "PER_mention_correct",
        "PER_mention_total",
        "PER_mention_accuracy",
        "LOC_mention_correct",
        "LOC_mention_total",
        "LOC_mention_accuracy",
        "ORG_mention_correct",
        "ORG_mention_total",
        "ORG_mention_accuracy",
        "DATE_mention_correct",
        "DATE_mention_total",
        "DATE_mention_accuracy",
        "model_loading_seconds",
        "audio_processing_seconds",
        "mean_audio_processing_seconds",
        "real_time_factor",
        "load_peak_rss_mb",
        "load_peak_rss_delta_mb",
        "inference_peak_rss_mb",
        "inference_peak_rss_delta_mb",
    ]
    write_csv_atomic(summaries, fieldnames, config.stt_summary_path)
    write_jsonl_atomic(predictions, config.stt_predictions_path)
    return tuple(summaries)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/evaluation.yaml")
    parser.add_argument("--model", action="append", choices=("base", "small"), dest="models")
    parser.add_argument(
        "--allow-incomplete-dataset",
        action="store_true",
        help="Run a labelled smoke set while preserving a noncompliant dataset report.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads; normal evaluation is local-files-only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_args(argv)
    config = load_evaluation_config(arguments.config)
    selected = tuple(dict.fromkeys(arguments.models or ("base", "small")))
    try:
        run_evaluation(
            config,
            selected_models=selected,
            allow_incomplete_dataset=bool(arguments.allow_incomplete_dataset),
            allow_download=bool(arguments.allow_download),
        )
    except (OSError, RuntimeError, ValueError) as error:
        LOGGER.error("STT evaluation failed: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
