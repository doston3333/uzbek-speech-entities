from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from evaluation.dataset import load_dataset
from evaluation.prepare_recording_manifest import (
    build_recording_plan,
    load_prompts,
    load_speaker_profiles,
    write_recording_plan,
)
from evaluation.recording_collection import (
    RecordingCollectionError,
    audit_collection,
    import_recordings,
    load_checklist,
    write_checklist,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = PROJECT_ROOT / "data/private_test/prompts.jsonl"
SPEAKERS_PATH = PROJECT_ROOT / "data/private_test/speakers.example.json"


def _write_plan(directory: Path) -> tuple[Path, Path]:
    plan = build_recording_plan(load_prompts(PROMPTS_PATH), load_speaker_profiles(SPEAKERS_PATH))
    metadata = directory / "metadata.jsonl"
    checklist = directory / "recording_checklist.csv"
    write_recording_plan(
        plan,
        metadata_path=metadata,
        checklist_path=checklist,
        summary_path=directory / "summary.json",
    )
    return metadata, checklist


def _confirm_consent(metadata: Path, checklist: Path, recording_id: str) -> None:
    dataset = load_dataset(metadata, require_files=False)
    entries = load_checklist(checklist, dataset)
    entries[recording_id] = replace(entries[recording_id], consent_confirmed=True)
    write_checklist(checklist, entries)


def _write_audio(
    path: Path, *, duration: float = 6.0, amplitude: float = 0.2, stereo: bool = True
) -> None:
    sample_rate = 16_000
    time = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 220 * time)
    samples = np.column_stack((signal, signal * 0.8)) if stereo else signal
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def test_empty_collection_audit_is_honest_progress_not_an_error(tmp_path: Path) -> None:
    metadata, checklist_path = _write_plan(tmp_path)
    dataset = load_dataset(metadata, require_files=False)
    checklist = load_checklist(checklist_path, dataset)

    progress = audit_collection(dataset, checklist)

    assert progress.status == "collection_in_progress"
    assert progress.planned_count == 150
    assert progress.audio_present_count == 0
    assert progress.ready_for_evaluation_count == 0
    assert progress.remaining_count == 150
    assert progress.issues == ()


def test_import_requires_consent_and_preserves_source(tmp_path: Path) -> None:
    metadata, checklist = _write_plan(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / "speaker-01-prompt-01.wav"
    _write_audio(source)

    with pytest.raises(RecordingCollectionError, match="explicit consent"):
        import_recordings(
            metadata_path=metadata,
            checklist_path=checklist,
            raw_directory=raw,
        )

    assert source.is_file()
    assert not (tmp_path / "audio/speaker-01-prompt-01.wav").exists()


def test_import_creates_canonical_wav_and_marks_recorded(tmp_path: Path) -> None:
    metadata, checklist_path = _write_plan(tmp_path)
    recording_id = "speaker-01-prompt-01"
    _confirm_consent(metadata, checklist_path, recording_id)
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / f"{recording_id}.wav"
    _write_audio(source, stereo=True)

    result = import_recordings(
        metadata_path=metadata,
        checklist_path=checklist_path,
        raw_directory=raw,
    )

    destination = tmp_path / f"audio/{recording_id}.wav"
    info = sf.info(destination)
    assert result.imported_count == 1
    assert result.replaced_count == 0
    assert result.already_present_count == 0
    assert result.progress.valid_audio_count == 1
    assert result.progress.ready_for_evaluation_count == 0
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.samplerate == 16_000
    assert info.channels == 1
    assert info.duration == pytest.approx(6.0)
    assert source.is_file()

    dataset = load_dataset(metadata, require_files=False)
    entries = load_checklist(checklist_path, dataset)
    assert entries[recording_id].consent_confirmed
    assert entries[recording_id].recorded
    assert not entries[recording_id].transcript_reviewed
    assert not entries[recording_id].entity_spans_reviewed


@pytest.mark.parametrize("duration", [4.0, 21.0])
def test_import_rejects_audio_outside_required_duration(tmp_path: Path, duration: float) -> None:
    metadata, checklist = _write_plan(tmp_path)
    recording_id = "speaker-01-prompt-01"
    _confirm_consent(metadata, checklist, recording_id)
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_audio(raw / f"{recording_id}.wav", duration=duration, stereo=False)

    with pytest.raises(RecordingCollectionError, match="duration must be 5-20 seconds"):
        import_recordings(
            metadata_path=metadata,
            checklist_path=checklist,
            raw_directory=raw,
        )

    assert not (tmp_path / f"audio/{recording_id}.wav").exists()


def test_import_is_idempotent_without_overwrite_and_replaces_only_when_explicit(
    tmp_path: Path,
) -> None:
    metadata, checklist = _write_plan(tmp_path)
    recording_id = "speaker-01-prompt-01"
    _confirm_consent(metadata, checklist, recording_id)
    raw = tmp_path / "raw"
    raw.mkdir()
    source = raw / f"{recording_id}.wav"
    _write_audio(source, amplitude=0.1)
    import_recordings(metadata_path=metadata, checklist_path=checklist, raw_directory=raw)
    destination = tmp_path / f"audio/{recording_id}.wav"
    original = destination.read_bytes()

    _write_audio(source, amplitude=0.7)
    skipped = import_recordings(metadata_path=metadata, checklist_path=checklist, raw_directory=raw)
    assert skipped.already_present_count == 1
    assert destination.read_bytes() == original

    replaced_result = import_recordings(
        metadata_path=metadata,
        checklist_path=checklist,
        raw_directory=raw,
        overwrite=True,
    )
    assert replaced_result.replaced_count == 1
    assert destination.read_bytes() != original


def test_import_rejects_unknown_and_duplicate_source_ids(tmp_path: Path) -> None:
    metadata, checklist = _write_plan(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_audio(raw / "not-planned.wav")
    with pytest.raises(RecordingCollectionError, match="not planned"):
        import_recordings(
            metadata_path=metadata,
            checklist_path=checklist,
            raw_directory=raw,
        )

    (raw / "not-planned.wav").unlink()
    recording_id = "speaker-01-prompt-01"
    _write_audio(raw / f"{recording_id}.wav")
    _write_audio(raw / f"{recording_id}.flac")
    with pytest.raises(RecordingCollectionError, match="multiple source files"):
        import_recordings(
            metadata_path=metadata,
            checklist_path=checklist,
            raw_directory=raw,
        )


def test_audit_reports_checklist_file_inconsistency(tmp_path: Path) -> None:
    metadata, checklist_path = _write_plan(tmp_path)
    dataset = load_dataset(metadata, require_files=False)
    entries = load_checklist(checklist_path, dataset)
    recording_id = "speaker-01-prompt-01"
    entries[recording_id] = replace(
        entries[recording_id],
        consent_confirmed=True,
        recorded=True,
        transcript_reviewed=True,
    )

    progress = audit_collection(dataset, entries)

    assert progress.status == "inconsistent"
    assert any("audio is missing" in issue for issue in progress.issues)
