from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uzbek_speech_entities.ner.model_selection import (
    INFERENCE_ARTIFACTS,
    PROVISIONAL_LIMITATION,
    RUN_ARTIFACTS,
    build_comparison_report,
    load_run,
    promote_selected_run,
    select_run,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _metrics(
    prefix: str, macro: float = 0.8, overall: float = 0.7, runtime: float = 2.0
) -> dict[str, float]:
    return {
        f"{prefix}LOC_f1": 0.7,
        f"{prefix}ORG_f1": 0.7,
        f"{prefix}PER_f1": 0.7,
        f"{prefix}TEMPORAL_f1": 0.7,
        f"{prefix}four_class_macro_f1": macro,
        f"{prefix}overall_f1": overall,
        f"{prefix}token_accuracy": 0.9,
        **({f"{prefix}runtime": runtime} if prefix else {}),
    }


def _make_run(
    base: Path,
    name: str,
    *,
    macro: float = 0.8,
    overall: float = 0.7,
    runtime: float = 2.0,
    weakest: float = 0.7,
    test_overall: float = 0.5,
) -> Path:
    root = base / name
    checkpoint = root / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    for file_name in INFERENCE_ARTIFACTS:
        (checkpoint / file_name).write_text(f"{name}-{file_name}", encoding="utf-8")
    validation = _metrics("eval_", macro, overall, runtime)
    for entity_type in ("PER", "LOC", "ORG", "TEMPORAL"):
        validation[f"eval_{entity_type}_f1"] = weakest
    testing = _metrics("")
    testing["overall_f1"] = test_overall
    _write_json(
        root / "best_checkpoint.json",
        {
            "checkpoint": "checkpoint-1",
            "metric_for_best_model": "overall_f1",
            "metric_value": overall,
        },
    )
    _write_json(root / "eval_metrics.json", validation)
    _write_json(root / "evaluation_test_metrics.json", testing)
    _write_json(
        root / "run_metadata.json",
        {"best_checkpoint": "checkpoint-1", "device": "cpu", "duration_seconds": 12.0},
    )
    for file_name in RUN_ARTIFACTS:
        (root / file_name).write_text("{}", encoding="utf-8")
    return root


def _selection(base: Path, **augmented: float) -> object:
    clean = load_run("clean", _make_run(base, "clean"))
    augmented_run = load_run("augmented", _make_run(base, "augmented", **augmented))
    return select_run(clean, augmented_run)


def _finalization_evidence(
    *, selected: str = "clean", clean_macro: float = 0.9, augmented_macro: float = 0.8
) -> dict[str, object]:
    def candidate(macro: float) -> dict[str, object]:
        return {
            "end_to_end_four_class_macro_f1": macro,
            "end_to_end_per_class_f1": {
                "PER": macro,
                "LOC": macro,
                "ORG": macro,
                "DATE": macro,
            },
            "clean_text_entity_f1": macro,
            "inference_latency_ms": 20.0,
            "stability_seed_four_class_macro_f1": {"42": macro, "43": macro - 0.01},
        }

    return {
        "candidates": {
            "clean": candidate(clean_macro),
            "augmented": candidate(augmented_macro),
        },
        "selected_run": selected,
    }


def test_augmented_wins_on_primary_validation_metric(tmp_path: Path) -> None:
    selection = _selection(tmp_path, macro=0.81)

    assert selection.selected.name == "augmented"


def test_clean_wins_exact_validation_tie_conservatively(tmp_path: Path) -> None:
    selection = _selection(tmp_path)

    assert selection.selected.name == "clean"


def test_weakest_class_then_overall_f1_break_validation_ties(tmp_path: Path) -> None:
    weakest_winner = _selection(tmp_path / "weakest", weakest=0.71)
    overall_winner = _selection(tmp_path / "overall", overall=0.71)

    assert weakest_winner.selected.name == "augmented"
    assert overall_winner.selected.name == "augmented"


def test_lower_runtime_breaks_remaining_validation_tie(tmp_path: Path) -> None:
    selection = _selection(tmp_path, runtime=1.9)

    assert selection.selected.name == "augmented"


def test_test_metrics_cannot_change_selection(tmp_path: Path) -> None:
    selection = _selection(tmp_path, test_overall=1.0)

    assert selection.selected.name == "clean"


def test_checkpoint_pointer_cannot_escape_run_root(tmp_path: Path) -> None:
    root = _make_run(tmp_path, "clean")
    _write_json(
        root / "best_checkpoint.json",
        {
            "checkpoint": "../outside",
            "metric_for_best_model": "overall_f1",
            "metric_value": 0.7,
        },
    )

    with pytest.raises(ValueError, match="escapes"):
        load_run("clean", root)


def test_checkpoint_pointer_metric_must_match_validation_artifact(tmp_path: Path) -> None:
    root = _make_run(tmp_path, "clean")
    pointer_path = root / "best_checkpoint.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["metric_value"] = 0.6
    _write_json(pointer_path, pointer)

    with pytest.raises(ValueError, match="does not match"):
        load_run("clean", root)


@pytest.mark.parametrize(
    ("artifact", "metric", "value"),
    [
        ("eval_metrics.json", "eval_PER_f1", None),
        ("eval_metrics.json", "eval_LOC_f1", float("nan")),
        ("evaluation_test_metrics.json", "overall_f1", 1.1),
    ],
)
def test_missing_nan_and_out_of_range_metrics_are_rejected(
    tmp_path: Path, artifact: str, metric: str, value: float | None
) -> None:
    root = _make_run(tmp_path, "clean")
    contents = json.loads((root / artifact).read_text(encoding="utf-8"))
    if value is None:
        del contents[metric]
    else:
        contents[metric] = value
    _write_json(root / artifact, contents)

    with pytest.raises(ValueError):
        load_run("clean", root)


def test_report_records_provisional_limitation_deltas_and_test_only_role(tmp_path: Path) -> None:
    selection = _selection(tmp_path, macro=0.81, test_overall=0.4)
    report = build_comparison_report(selection)

    assert report["selected_run"] == "augmented"
    assert report["provisional"] is True
    assert report["limitation"] == PROVISIONAL_LIMITATION
    assert report["test_metrics_role"].startswith("report-only")
    assert report["deltas_augmented_minus_clean"]["validation"]["eval_four_class_macro_f1"] > 0
    assert report["deltas_augmented_minus_clean"]["test"]["overall_f1"] < 0
    assert "eval_four_class_macro_f1" in report["improvements"]["validation"]
    assert "overall_f1" in report["regressions"]["test"]


def test_finalization_evidence_can_replace_provisional_validation_winner(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    evidence = _finalization_evidence(selected="augmented", clean_macro=0.8, augmented_macro=0.9)

    report = build_comparison_report(selection, finalization_evidence=evidence)

    assert report["selected_run"] == "augmented"
    assert report["provisional"] is False
    assert report["limitation"] is None


def test_promotion_refuses_missing_finalization_evidence(tmp_path: Path) -> None:
    selection = _selection(tmp_path)

    with pytest.raises(ValueError, match="requires finalization evidence"):
        promote_selected_run(selection, tmp_path / "final", tmp_path / "report.json")


def test_promotion_copies_inference_only_and_records_model_sha(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    source = selection.selected.checkpoint
    (source / "optimizer.pt").write_text("training state", encoding="utf-8")
    final = promote_selected_run(
        selection,
        tmp_path / "final",
        tmp_path / "report.json",
        finalization_evidence=_finalization_evidence(),
    )

    assert {path.name for path in final.iterdir()} == {
        *INFERENCE_ARTIFACTS,
        *RUN_ARTIFACTS,
        "selection.json",
    }
    manifest = json.loads((final / "selection.json").read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256((source / "model.safetensors").read_bytes()).hexdigest()
    assert manifest["model_safetensors_sha256"] == expected_sha
    assert manifest["source_run"] == "clean"
    assert manifest["provisional"] is False
    assert manifest["limitation"] is None


def test_promotion_refuses_existing_final_without_overwrite(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    final = tmp_path / "final"
    final.mkdir()
    (final / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        promote_selected_run(
            selection,
            final,
            tmp_path / "report.json",
            finalization_evidence=_finalization_evidence(),
        )
    assert (final / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_overwrite_replaces_final_only_after_complete_promotion(tmp_path: Path) -> None:
    selection = _selection(tmp_path)
    final = tmp_path / "final"
    final.mkdir()
    (final / "obsolete.txt").write_text("old", encoding="utf-8")

    promoted = promote_selected_run(
        selection,
        final,
        tmp_path / "report.json",
        finalization_evidence=_finalization_evidence(),
        overwrite_final=True,
    )

    assert promoted == final.resolve()
    assert (final / "model.safetensors").is_file()
    assert not (final / "obsolete.txt").exists()
