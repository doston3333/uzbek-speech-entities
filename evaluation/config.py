"""Validated paths and model identifiers for Phase 8 evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from uzbek_speech_entities.config import resolve_project_path


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    metadata_path: Path
    compliance_report_path: Path
    require_compliance: bool
    stt_model_ids: Mapping[str, str]
    stt_summary_path: Path
    stt_predictions_path: Path
    stt_local_files_only: bool
    clean_ner_run_path: Path
    augmented_ner_run_path: Path
    pipeline_summary_path: Path
    pipeline_predictions_path: Path
    pipeline_selection_path: Path
    error_summary_path: Path
    error_records_path: Path
    primary_ablation: str


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"evaluation configuration {location} must be a mapping")
    return value


def _string(values: Mapping[str, Any], key: str, location: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evaluation configuration {location}.{key} must be non-empty text")
    return value


def _boolean(values: Mapping[str, Any], key: str, location: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"evaluation configuration {location}.{key} must be boolean")
    return value


def _path(values: Mapping[str, Any], key: str, location: str) -> Path:
    return Path(resolve_project_path(_string(values, key, location)))


def load_evaluation_config(path: str | Path = "configs/evaluation.yaml") -> EvaluationConfig:
    """Load the evaluation configuration without accepting ambiguous defaults."""
    config_path = Path(resolve_project_path(path))
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"could not read evaluation configuration: {config_path}") from error
    root = _mapping(raw, "root")
    dataset = _mapping(root.get("dataset"), "dataset")
    stt = _mapping(root.get("stt"), "stt")
    model_ids_raw = _mapping(stt.get("model_ids"), "stt.model_ids")
    model_ids = {
        key: _string(model_ids_raw, key, "stt.model_ids") for key in ("base", "small")
    }
    pipeline = _mapping(root.get("pipeline"), "pipeline")
    reports = _mapping(root.get("reports"), "reports")
    primary_ablation = _string(reports, "primary_ablation", "reports")
    if primary_ablation not in set("ABCDEFGH"):
        raise ValueError("evaluation configuration reports.primary_ablation must be A-H")
    return EvaluationConfig(
        metadata_path=_path(dataset, "metadata_path", "dataset"),
        compliance_report_path=_path(dataset, "compliance_report_path", "dataset"),
        require_compliance=_boolean(dataset, "require_compliance", "dataset"),
        stt_model_ids=model_ids,
        stt_summary_path=_path(stt, "summary_path", "stt"),
        stt_predictions_path=_path(stt, "predictions_path", "stt"),
        stt_local_files_only=_boolean(stt, "local_files_only", "stt"),
        clean_ner_run_path=_path(pipeline, "clean_ner_run_path", "pipeline"),
        augmented_ner_run_path=_path(pipeline, "augmented_ner_run_path", "pipeline"),
        pipeline_summary_path=_path(pipeline, "summary_path", "pipeline"),
        pipeline_predictions_path=_path(pipeline, "predictions_path", "pipeline"),
        pipeline_selection_path=_path(pipeline, "selection_path", "pipeline"),
        error_summary_path=_path(reports, "error_summary_path", "reports"),
        error_records_path=_path(reports, "error_records_path", "reports"),
        primary_ablation=primary_ablation,
    )
