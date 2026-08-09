from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluation.dataset import (
    REQUIRED_CONDITIONS,
    REQUIRED_CONTENT_COVERAGE,
    build_compliance_report,
    load_dataset,
)
from evaluation.prepare_recording_manifest import (
    RecordingPlan,
    RecordingPlanError,
    build_recording_plan,
    load_prompts,
    load_speaker_profiles,
    write_recording_plan,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = PROJECT_ROOT / "data/private_test/prompts.jsonl"
SPEAKERS_PATH = PROJECT_ROOT / "data/private_test/speakers.example.json"


def _committed_plan() -> RecordingPlan:
    return build_recording_plan(load_prompts(PROMPTS_PATH), load_speaker_profiles(SPEAKERS_PATH))


def test_committed_collection_pack_proves_planned_targets() -> None:
    prompts = load_prompts(PROMPTS_PATH)
    speakers = load_speaker_profiles(SPEAKERS_PATH)
    plan = build_recording_plan(prompts, speakers)

    assert len(prompts) == 30
    assert len(speakers) == 5
    assert len(plan.metadata_rows) == 150
    assert len({row["id"] for row in plan.metadata_rows}) == 150
    assert len({row["file"] for row in plan.metadata_rows}) == 150
    assert plan.summary["status"] == "planned_not_recorded"
    assert plan.summary["recorded_count"] == 0
    assert plan.summary["entity_counts"] == {
        "PER": 150,
        "LOC": 150,
        "ORG": 150,
        "DATE": 150,
    }
    condition_coverage = plan.summary["condition_coverage"]
    content_coverage = plan.summary["content_coverage"]
    assert isinstance(condition_coverage, list)
    assert isinstance(content_coverage, list)
    assert set(condition_coverage) == set(REQUIRED_CONDITIONS)
    assert set(content_coverage) == set(REQUIRED_CONTENT_COVERAGE)


def test_written_plan_matches_dataset_schema_without_claiming_audio(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    checklist = tmp_path / "recording_checklist.csv"
    summary = tmp_path / "collection_plan.json"
    write_recording_plan(
        _committed_plan(),
        metadata_path=metadata,
        checklist_path=checklist,
        summary_path=summary,
    )

    dataset = load_dataset(metadata, require_files=False)
    report = build_compliance_report(dataset, inspect_files=False)
    assert report.recording_count == 150
    assert report.speaker_count == 5
    assert report.entity_counts == {"PER": 150, "LOC": 150, "ORG": 150, "DATE": 150}
    assert report.issues == ("audio durations were not inspected",)
    assert json.loads(summary.read_text(encoding="utf-8"))["recorded_count"] == 0

    with checklist.open(encoding="utf-8", newline="") as stream:
        checklist_rows = list(csv.DictReader(stream))
    assert len(checklist_rows) == 150
    for row in checklist_rows:
        assert row["consent_confirmed"] == "False"
        assert row["recorded"] == "False"
        assert row["transcript_reviewed"] == "False"
        assert row["entity_spans_reviewed"] == "False"


def test_writer_refuses_to_overwrite_collection_files(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    checklist = tmp_path / "checklist.csv"
    summary = tmp_path / "summary.json"
    plan = _committed_plan()
    write_recording_plan(
        plan,
        metadata_path=metadata,
        checklist_path=checklist,
        summary_path=summary,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_recording_plan(
            plan,
            metadata_path=metadata,
            checklist_path=checklist,
            summary_path=summary,
        )


def test_prompt_pack_rejects_repeated_entity_surface(tmp_path: Path) -> None:
    first_prompt = json.loads(PROMPTS_PATH.read_text(encoding="utf-8").splitlines()[0])
    first_prompt["text"] += " Akmal Karimov uchrashuvni yakunlaydi."
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(json.dumps(first_prompt, ensure_ascii=False) + "\n", encoding="utf-8")

    with pytest.raises(RecordingPlanError, match="must occur exactly once"):
        load_prompts(prompt_path)


def test_speaker_plan_requires_five_profiles(tmp_path: Path) -> None:
    speakers = json.loads(SPEAKERS_PATH.read_text(encoding="utf-8"))[:4]
    speaker_path = tmp_path / "speakers.json"
    speaker_path.write_text(json.dumps(speakers), encoding="utf-8")

    with pytest.raises(RecordingPlanError, match="at least five"):
        load_speaker_profiles(speaker_path)
