from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset import (
    REQUIRED_CONDITIONS,
    REQUIRED_CONTENT_COVERAGE,
    DatasetValidationError,
    build_compliance_report,
    load_dataset,
)


def _record(*, sample_id: str = "audio-001", **changes: object) -> dict[str, object]:
    record: dict[str, object] = {
        "id": sample_id,
        "file": "audio/example.wav",
        "gold_transcript": "Akmal Toshkentga bordi.",
        "entities": [{"text": "Akmal", "label": "PER", "start": 0, "end": 5}],
        "speaker_id": "speaker-01",
        "conditions": ["quiet"],
    }
    record.update(changes)
    return record


def _write_metadata(path: Path, records: list[dict[str, object]]) -> Path:
    metadata = path / "metadata.jsonl"
    metadata.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return metadata


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"id": "only"}, "schema mismatch"),
        (_record(file="../escape.wav"), "escapes"),
        (
            _record(entities=[{"text": "wrong", "label": "PER", "start": 0, "end": 5}]),
            "does not match",
        ),
        (
            _record(
                entities=[
                    {"text": "Akmal", "label": "PER", "start": 0, "end": 5},
                    {"text": "mal", "label": "LOC", "start": 2, "end": 5},
                ]
            ),
            "ordered and non-overlapping",
        ),
    ],
)
def test_loader_rejects_invalid_schema_and_spans(
    tmp_path: Path, record: dict[str, object], message: str
) -> None:
    with pytest.raises(DatasetValidationError, match=message):
        load_dataset(_write_metadata(tmp_path, [record]))


def test_loader_rejects_malformed_json_duplicates_and_missing_file(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="valid JSON"):
        load_dataset(metadata)

    metadata = _write_metadata(tmp_path, [_record(), _record(sample_id="audio-001")])
    with pytest.raises(DatasetValidationError, match="duplicate id"):
        load_dataset(metadata)
    with pytest.raises(DatasetValidationError, match="audio file does not exist"):
        load_dataset(_write_metadata(tmp_path, [_record()]), require_files=True)


def test_compliance_report_never_fabricates_missing_coverage(tmp_path: Path) -> None:
    dataset = load_dataset(_write_metadata(tmp_path, [_record()]))
    report = build_compliance_report(dataset, inspect_files=False)

    assert not report.compliant
    assert report.recording_count == 1
    assert report.entity_counts == {"PER": 1, "LOC": 0, "ORG": 0, "DATE": 0}
    assert set(REQUIRED_CONDITIONS) - set(report.condition_coverage)
    assert set(REQUIRED_CONTENT_COVERAGE) - set(report.content_coverage)
    assert any("audio durations were not inspected" in issue for issue in report.issues)
