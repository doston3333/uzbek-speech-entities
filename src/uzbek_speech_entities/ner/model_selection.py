"""Offline, validation-only comparison and safe inference-model promotion for NER runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRIMARY_CLASSES = ("PER", "LOC", "ORG", "TEMPORAL")
FINALIZATION_CLASSES = ("PER", "LOC", "ORG", "DATE")
VALIDATION_METRICS = (
    "eval_overall_f1",
    "eval_four_class_macro_f1",
    "eval_token_accuracy",
    "eval_PER_f1",
    "eval_LOC_f1",
    "eval_ORG_f1",
    "eval_TEMPORAL_f1",
)
TEST_METRICS = tuple(metric.removeprefix("eval_") for metric in VALIDATION_METRICS)
INFERENCE_ARTIFACTS = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
RUN_ARTIFACTS = ("labels.json", "package_versions.json")
PROVISIONAL_LIMITATION = (
    "Selection is provisional: end-to-end STT macro F1 and cross-seed training stability "
    "are deferred."
)
RANKING_CRITERIA = (
    "higher eval_four_class_macro_f1",
    "higher minimum of eval_PER_f1, eval_LOC_f1, eval_ORG_f1, and eval_TEMPORAL_f1",
    "higher eval_overall_f1",
    "lower positive eval_runtime",
    "exact ties prefer clean conservatively",
)
FINAL_RANKING_CRITERIA = (
    "higher end_to_end_four_class_macro_f1",
    "higher weakest end-to-end public-class F1",
    "higher clean_text_entity_f1",
    "lower inference_latency_ms",
    "higher worst multi-seed four-class macro F1",
    "lower multi-seed four-class macro F1 range",
    "exact ties prefer clean conservatively",
)


@dataclass(frozen=True)
class NerRun:
    """Validated artifacts from one auditable NER training run."""

    name: str
    root: Path
    checkpoint: Path
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    metadata: dict[str, Any]

    @property
    def ranking_values(self) -> dict[str, float]:
        """Return the validation-only values used in deterministic ranking."""
        weakest_class = min(
            self.validation_metrics[f"eval_{entity_type}_f1"] for entity_type in PRIMARY_CLASSES
        )
        return {
            "four_class_macro_f1": self.validation_metrics["eval_four_class_macro_f1"],
            "weakest_primary_class_f1": weakest_class,
            "overall_f1": self.validation_metrics["eval_overall_f1"],
            "runtime_seconds": self.validation_metrics["eval_runtime"],
        }


@dataclass(frozen=True)
class Selection:
    """A validation-only selected run, intentionally independent of test metrics."""

    clean: NerRun
    augmented: NerRun
    selected: NerRun


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _finite_number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return number


def _validated_metrics(
    metrics: dict[str, Any], required: tuple[str, ...], artifact: Path, *, validation: bool
) -> dict[str, float]:
    validated: dict[str, float] = {}
    for metric in required:
        if metric not in metrics:
            raise ValueError(f"missing metric {metric} in {artifact}")
        validated[metric] = _finite_number(
            metrics[metric], f"{artifact}:{metric}", minimum=0.0, maximum=1.0
        )
    if validation:
        if "eval_runtime" not in metrics:
            raise ValueError(f"missing metric eval_runtime in {artifact}")
        validated["eval_runtime"] = _finite_number(
            metrics["eval_runtime"], f"{artifact}:eval_runtime", minimum=0.0, maximum=math.inf
        )
        if validated["eval_runtime"] <= 0:
            raise ValueError(f"{artifact}:eval_runtime must be positive")
    return validated


def _resolve_checkpoint(run_root: Path, pointer: dict[str, Any]) -> Path:
    raw_checkpoint = pointer.get("checkpoint")
    if not isinstance(raw_checkpoint, str) or not raw_checkpoint:
        raise ValueError(f"invalid checkpoint pointer in {run_root / 'best_checkpoint.json'}")
    relative_checkpoint = Path(raw_checkpoint)
    if relative_checkpoint.is_absolute():
        raise ValueError("checkpoint pointer must be relative to its run root")
    resolved_root = run_root.resolve()
    checkpoint = (resolved_root / relative_checkpoint).resolve()
    try:
        checkpoint.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("checkpoint pointer escapes its run root") from error
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
    return checkpoint


def _require_files(directory: Path, file_names: tuple[str, ...]) -> None:
    missing = [file_name for file_name in file_names if not (directory / file_name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts in {directory}: {', '.join(missing)}")


def load_run(name: str, run_root: Path) -> NerRun:
    """Load one run and validate all artifacts needed for comparison and promotion."""
    root = run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {root}")
    required = (
        "best_checkpoint.json",
        "eval_metrics.json",
        "evaluation_test_metrics.json",
        "run_metadata.json",
    )
    _require_files(root, required + RUN_ARTIFACTS)
    pointer = _read_json(root / "best_checkpoint.json")
    checkpoint = _resolve_checkpoint(root, pointer)
    _require_files(checkpoint, INFERENCE_ARTIFACTS)
    validation_metrics = _validated_metrics(
        _read_json(root / "eval_metrics.json"),
        VALIDATION_METRICS,
        root / "eval_metrics.json",
        validation=True,
    )
    if pointer.get("metric_for_best_model") != "overall_f1":
        raise ValueError(f"best checkpoint pointer must select overall_f1 in {root}")
    pointer_metric = _finite_number(
        pointer.get("metric_value"),
        f"{root / 'best_checkpoint.json'}:metric_value",
        minimum=0.0,
        maximum=1.0,
    )
    if not math.isclose(
        pointer_metric,
        validation_metrics["eval_overall_f1"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"best checkpoint metric does not match validation metrics in {root}")
    test_metrics = _validated_metrics(
        _read_json(root / "evaluation_test_metrics.json"),
        TEST_METRICS,
        root / "evaluation_test_metrics.json",
        validation=False,
    )
    metadata = _read_json(root / "run_metadata.json")
    if not isinstance(metadata.get("device"), str) or not metadata["device"]:
        raise ValueError(f"run metadata must contain a device: {root / 'run_metadata.json'}")
    if "duration_seconds" not in metadata:
        raise ValueError(
            f"run metadata must contain duration_seconds: {root / 'run_metadata.json'}"
        )
    _finite_number(metadata["duration_seconds"], "duration_seconds", minimum=0.0, maximum=math.inf)
    if metadata.get("best_checkpoint") != pointer["checkpoint"]:
        raise ValueError(f"best checkpoint metadata does not match pointer in {root}")
    return NerRun(name, root, checkpoint, validation_metrics, test_metrics, metadata)


def select_run(clean: NerRun, augmented: NerRun) -> Selection:
    """Select by validation ranking alone; exact ties intentionally retain the clean run."""
    clean_values = clean.ranking_values
    augmented_values = augmented.ranking_values
    clean_score = (
        clean_values["four_class_macro_f1"],
        clean_values["weakest_primary_class_f1"],
        clean_values["overall_f1"],
        -clean_values["runtime_seconds"],
    )
    augmented_score = (
        augmented_values["four_class_macro_f1"],
        augmented_values["weakest_primary_class_f1"],
        augmented_values["overall_f1"],
        -augmented_values["runtime_seconds"],
    )
    return Selection(clean, augmented, augmented if augmented_score > clean_score else clean)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _finalization_candidate(evidence: Mapping[str, Any], run_name: str) -> dict[str, Any]:
    candidate = _mapping(evidence.get(run_name), f"finalization candidates.{run_name}")
    per_class = _mapping(
        candidate.get("end_to_end_per_class_f1"),
        f"finalization candidates.{run_name}.end_to_end_per_class_f1",
    )
    if set(per_class) != set(FINALIZATION_CLASSES):
        raise ValueError(
            f"finalization candidates.{run_name}.end_to_end_per_class_f1 must contain "
            f"exactly {FINALIZATION_CLASSES}"
        )
    per_class_f1 = {
        label: _finite_number(
            per_class[label],
            f"finalization candidates.{run_name}.end_to_end_per_class_f1.{label}",
            minimum=0.0,
            maximum=1.0,
        )
        for label in FINALIZATION_CLASSES
    }
    seed_values = _mapping(
        candidate.get("stability_seed_four_class_macro_f1"),
        f"finalization candidates.{run_name}.stability_seed_four_class_macro_f1",
    )
    if len(seed_values) < 2 or not all(
        isinstance(seed, str) and seed.strip() for seed in seed_values
    ):
        raise ValueError(f"finalization candidates.{run_name} requires at least two seed scores")
    seed_scores = {
        seed: _finite_number(
            value,
            f"finalization candidates.{run_name}.stability_seed_four_class_macro_f1.{seed}",
            minimum=0.0,
            maximum=1.0,
        )
        for seed, value in seed_values.items()
    }
    end_to_end_macro = _finite_number(
        candidate.get("end_to_end_four_class_macro_f1"),
        f"finalization candidates.{run_name}.end_to_end_four_class_macro_f1",
        minimum=0.0,
        maximum=1.0,
    )
    clean_text_f1 = _finite_number(
        candidate.get("clean_text_entity_f1"),
        f"finalization candidates.{run_name}.clean_text_entity_f1",
        minimum=0.0,
        maximum=1.0,
    )
    latency_ms = _finite_number(
        candidate.get("inference_latency_ms"),
        f"finalization candidates.{run_name}.inference_latency_ms",
        minimum=0.0,
        maximum=math.inf,
    )
    if latency_ms <= 0:
        raise ValueError(
            f"finalization candidates.{run_name}.inference_latency_ms must be positive"
        )
    stability_scores = tuple(seed_scores.values())
    return {
        "clean_text_entity_f1": clean_text_f1,
        "end_to_end_four_class_macro_f1": end_to_end_macro,
        "end_to_end_per_class_f1": per_class_f1,
        "inference_latency_ms": latency_ms,
        "stability_seed_four_class_macro_f1": seed_scores,
        "stability_min_four_class_macro_f1": min(stability_scores),
        "stability_range_four_class_macro_f1": max(stability_scores) - min(stability_scores),
        "weakest_end_to_end_class_f1": min(per_class_f1.values()),
    }


def _finalization_rank(values: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(values["end_to_end_four_class_macro_f1"]),
        float(values["weakest_end_to_end_class_f1"]),
        float(values["clean_text_entity_f1"]),
        -float(values["inference_latency_ms"]),
        float(values["stability_min_four_class_macro_f1"]),
        -float(values["stability_range_four_class_macro_f1"]),
    )


def finalize_selection(
    selection: Selection, evidence: Mapping[str, Any]
) -> tuple[Selection, dict[str, Any]]:
    """Validate final evidence for both candidates and derive the publishable winner."""
    candidates = _mapping(evidence.get("candidates"), "finalization candidates")
    if set(candidates) != {"clean", "augmented"}:
        raise ValueError("finalization candidates must contain exactly clean and augmented")
    normalized_candidates = {
        name: _finalization_candidate(candidates, name) for name in ("clean", "augmented")
    }
    clean_rank = _finalization_rank(normalized_candidates["clean"])
    augmented_rank = _finalization_rank(normalized_candidates["augmented"])
    winner_name = "augmented" if augmented_rank > clean_rank else "clean"
    if evidence.get("selected_run") != winner_name:
        raise ValueError("finalization selected_run does not match the evidence-derived ranking")
    winner = selection.augmented if winner_name == "augmented" else selection.clean
    normalized = {
        "candidates": normalized_candidates,
        "criteria": list(FINAL_RANKING_CRITERIA),
        "selected_run": winner_name,
    }
    return Selection(selection.clean, selection.augmented, winner), normalized


def _run_report(run: NerRun) -> dict[str, Any]:
    return {
        "checkpoint": str(run.checkpoint),
        "device": run.metadata["device"],
        "duration_seconds": run.metadata["duration_seconds"],
        "run_root": str(run.root),
        "runtime_metadata": run.metadata,
        "test_metrics": run.test_metrics,
        "validation_metrics": run.validation_metrics,
    }


def _metric_deltas(clean: dict[str, float], augmented: dict[str, float]) -> dict[str, float]:
    return {metric: augmented[metric] - clean[metric] for metric in clean}


def _changes(deltas: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    improvements = {metric: value for metric, value in deltas.items() if value > 0}
    regressions = {metric: value for metric, value in deltas.items() if value < 0}
    return improvements, regressions


def build_comparison_report(
    selection: Selection, *, finalization_evidence: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Build a serializable report, retaining test metrics strictly as report-only evidence."""
    report_selection = selection
    normalized_finalization: dict[str, Any] | None = None
    if finalization_evidence is not None:
        report_selection, normalized_finalization = finalize_selection(
            selection, finalization_evidence
        )
    validation_deltas = _metric_deltas(
        selection.clean.validation_metrics, selection.augmented.validation_metrics
    )
    test_deltas = _metric_deltas(selection.clean.test_metrics, selection.augmented.test_metrics)
    validation_improvements, validation_regressions = _changes(validation_deltas)
    test_improvements, test_regressions = _changes(test_deltas)
    return {
        "clean": _run_report(selection.clean),
        "augmented": _run_report(selection.augmented),
        "deltas_augmented_minus_clean": {
            "validation": validation_deltas,
            "test": test_deltas,
        },
        "improvements": {"validation": validation_improvements, "test": test_improvements},
        "limitation": PROVISIONAL_LIMITATION if normalized_finalization is None else None,
        "provisional": normalized_finalization is None,
        "regressions": {"validation": validation_regressions, "test": test_regressions},
        "selected_checkpoint": str(report_selection.selected.checkpoint),
        "selected_run": report_selection.selected.name,
        "selection_criteria": list(
            RANKING_CRITERIA if normalized_finalization is None else FINAL_RANKING_CRITERIA
        ),
        "selection_ranking_values": {
            "augmented": selection.augmented.ranking_values,
            "clean": selection.clean.ranking_values,
        },
        "test_metrics_role": "report-only; test metrics do not participate in model selection",
        **(
            {"finalization_evidence": normalized_finalization}
            if normalized_finalization is not None
            else {}
        ),
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically write JSON within its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_manifest(
    selection: Selection,
    report_path: Path,
    model_sha256: str,
    finalization_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "finalization_evidence": finalization_evidence,
        "limitation": None,
        "model_safetensors_sha256": model_sha256,
        "provisional": False,
        "report_path": str(report_path),
        "selection_metrics": selection.selected.ranking_values,
        "source_checkpoint": str(selection.selected.checkpoint),
        "source_run": selection.selected.name,
    }


def promote_selected_run(
    selection: Selection,
    final_root: Path,
    report_path: Path,
    *,
    finalization_evidence: Mapping[str, Any] | None = None,
    overwrite_final: bool = False,
) -> Path:
    """Atomically promote only inference artifacts, preserving an old final model on failure."""
    if finalization_evidence is None:
        raise ValueError("final model promotion requires finalization evidence")
    finalized, normalized_finalization = finalize_selection(selection, finalization_evidence)
    final = final_root.resolve()
    if final.exists() and not overwrite_final:
        raise FileExistsError(f"refusing to overwrite existing final model: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=final.parent))
    backup: Path | None = None
    promoted = False
    try:
        for file_name in INFERENCE_ARTIFACTS:
            shutil.copy2(finalized.selected.checkpoint / file_name, temporary / file_name)
        for file_name in RUN_ARTIFACTS:
            shutil.copy2(finalized.selected.root / file_name, temporary / file_name)
        model_sha256 = _sha256(temporary / "model.safetensors")
        write_json_atomic(
            temporary / "selection.json",
            _selection_manifest(
                finalized,
                report_path,
                model_sha256,
                normalized_finalization,
            ),
        )
        if final.exists():
            backup = final.with_name(f".{final.name}.backup-{uuid.uuid4().hex}")
            os.replace(final, backup)
        try:
            os.replace(temporary, final)
            promoted = True
        except OSError:
            if backup is not None and backup.exists():
                if final.exists():
                    shutil.rmtree(final)
                os.replace(backup, final)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return final
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and backup.exists() and not promoted and not final.exists():
            os.replace(backup, final)
