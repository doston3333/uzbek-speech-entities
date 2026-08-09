"""Consent-gated import and progress auditing for private evaluation recordings."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

from evaluation.dataset import (
    DatasetValidationError,
    EvaluationDataset,
    EvaluationSample,
    load_dataset,
)
from evaluation.io_utils import write_csv_atomic, write_json_atomic
from uzbek_speech_entities.audio.loader import AudioDecodingError, decode_audio
from uzbek_speech_entities.audio.preprocessing import ProcessedAudio, preprocess_audio
from uzbek_speech_entities.audio.validation import AudioValidationError
from uzbek_speech_entities.config import resolve_project_path

LOGGER = logging.getLogger(__name__)
CHECKLIST_FIELDS: Final[tuple[str, ...]] = (
    "recording_id",
    "speaker_id",
    "prompt_id",
    "audio_file",
    "consent_confirmed",
    "recorded",
    "transcript_reviewed",
    "entity_spans_reviewed",
)
SUPPORTED_SOURCE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".wav", ".mp3", ".m4a", ".webm", ".flac"}
)
TARGET_SAMPLE_RATE: Final = 16_000
MIN_DURATION_SECONDS: Final = 5.0
MAX_DURATION_SECONDS: Final = 20.0
MAX_SOURCE_BYTES: Final = 25 * 1024 * 1024


class RecordingCollectionError(ValueError):
    """Raised when collection state is unsafe, ambiguous, or inconsistent."""


@dataclass(frozen=True, slots=True)
class ChecklistEntry:
    recording_id: str
    speaker_id: str
    prompt_id: str
    audio_file: str
    consent_confirmed: bool
    recorded: bool
    transcript_reviewed: bool
    entity_spans_reviewed: bool

    def to_row(self) -> Mapping[str, object]:
        return {
            "recording_id": self.recording_id,
            "speaker_id": self.speaker_id,
            "prompt_id": self.prompt_id,
            "audio_file": self.audio_file,
            "consent_confirmed": self.consent_confirmed,
            "recorded": self.recorded,
            "transcript_reviewed": self.transcript_reviewed,
            "entity_spans_reviewed": self.entity_spans_reviewed,
        }


@dataclass(frozen=True, slots=True)
class CollectionProgress:
    status: str
    planned_count: int
    consented_count: int
    audio_present_count: int
    valid_audio_count: int
    recorded_checklist_count: int
    transcript_reviewed_count: int
    entity_spans_reviewed_count: int
    ready_for_evaluation_count: int
    remaining_count: int
    issues: tuple[str, ...]

    def to_dict(self) -> Mapping[str, object]:
        return {
            "status": self.status,
            "planned_count": self.planned_count,
            "consented_count": self.consented_count,
            "audio_present_count": self.audio_present_count,
            "valid_audio_count": self.valid_audio_count,
            "recorded_checklist_count": self.recorded_checklist_count,
            "transcript_reviewed_count": self.transcript_reviewed_count,
            "entity_spans_reviewed_count": self.entity_spans_reviewed_count,
            "ready_for_evaluation_count": self.ready_for_evaluation_count,
            "remaining_count": self.remaining_count,
            "issues": list(self.issues),
        }


@dataclass(frozen=True, slots=True)
class ImportResult:
    imported_count: int
    replaced_count: int
    already_present_count: int
    recording_ids: tuple[str, ...]
    progress: CollectionProgress

    def to_dict(self) -> Mapping[str, object]:
        return {
            "imported_count": self.imported_count,
            "replaced_count": self.replaced_count,
            "already_present_count": self.already_present_count,
            "recording_ids": list(self.recording_ids),
            "progress": self.progress.to_dict(),
        }


def _required_cell(row: Mapping[str, str | None], field: str, location: str) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise RecordingCollectionError(f"{location}.{field} must be non-empty")
    return value.strip()


def _boolean_cell(row: Mapping[str, str | None], field: str, location: str) -> bool:
    value = _required_cell(row, field, location).lower()
    if value not in {"true", "false"}:
        raise RecordingCollectionError(f"{location}.{field} must be true or false")
    return value == "true"


def _relative_audio_file(sample: EvaluationSample, dataset: EvaluationDataset) -> str:
    return sample.file.relative_to(dataset.metadata_path.parent).as_posix()


def load_checklist(path: str | Path, dataset: EvaluationDataset) -> dict[str, ChecklistEntry]:
    """Load an exact-schema checklist and prove it matches the metadata one-to-one."""
    source = Path(path)
    try:
        stream = source.open(encoding="utf-8", newline="")
    except OSError as error:
        raise RecordingCollectionError(f"could not read recording checklist: {source}") from error
    with stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CHECKLIST_FIELDS:
            raise RecordingCollectionError(
                "recording checklist header must exactly match: " + ", ".join(CHECKLIST_FIELDS)
            )
        entries: dict[str, ChecklistEntry] = {}
        for line_number, row in enumerate(reader, start=2):
            location = f"checklist line {line_number}"
            if None in row:
                raise RecordingCollectionError(f"{location} has unexpected extra columns")
            recording_id = _required_cell(row, "recording_id", location)
            if recording_id in entries:
                raise RecordingCollectionError(f"duplicate checklist recording_id: {recording_id}")
            entries[recording_id] = ChecklistEntry(
                recording_id=recording_id,
                speaker_id=_required_cell(row, "speaker_id", location),
                prompt_id=_required_cell(row, "prompt_id", location),
                audio_file=_required_cell(row, "audio_file", location),
                consent_confirmed=_boolean_cell(row, "consent_confirmed", location),
                recorded=_boolean_cell(row, "recorded", location),
                transcript_reviewed=_boolean_cell(row, "transcript_reviewed", location),
                entity_spans_reviewed=_boolean_cell(row, "entity_spans_reviewed", location),
            )

    samples = {sample.id: sample for sample in dataset.samples}
    missing = sorted(set(samples) - set(entries))
    unexpected = sorted(set(entries) - set(samples))
    if missing or unexpected:
        raise RecordingCollectionError(
            f"checklist/metadata IDs differ; missing={missing}, unexpected={unexpected}"
        )
    for recording_id, sample in samples.items():
        entry = entries[recording_id]
        if entry.speaker_id != sample.speaker_id:
            raise RecordingCollectionError(
                f"{recording_id}: checklist speaker_id does not match metadata"
            )
        if entry.audio_file != _relative_audio_file(sample, dataset):
            raise RecordingCollectionError(
                f"{recording_id}: checklist audio_file does not match metadata"
            )
    return entries


def write_checklist(path: str | Path, entries: Mapping[str, ChecklistEntry]) -> None:
    """Atomically preserve every checklist field in deterministic recording order."""
    rows = (entries[recording_id].to_row() for recording_id in sorted(entries))
    write_csv_atomic(rows, CHECKLIST_FIELDS, path)


def inspect_canonical_audio(path: str | Path) -> float:
    """Validate one evaluation file as a uniform 16 kHz mono PCM WAV."""
    source = Path(path)
    try:
        info = sf.info(source)
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise RecordingCollectionError(f"audio could not be inspected: {source}") from error
    duration = info.frames / info.samplerate if info.samplerate > 0 else 0.0
    problems: list[str] = []
    if info.format != "WAV":
        problems.append(f"format is {info.format or 'unknown'}, expected WAV")
    if info.subtype != "PCM_16":
        problems.append(f"subtype is {info.subtype or 'unknown'}, expected PCM_16")
    if info.samplerate != TARGET_SAMPLE_RATE:
        problems.append(f"sample rate is {info.samplerate}, expected {TARGET_SAMPLE_RATE}")
    if info.channels != 1:
        problems.append(f"channel count is {info.channels}, expected 1")
    if info.frames <= 0:
        problems.append("audio contains no frames")
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        problems.append(
            f"duration is {duration:.3f}s, expected "
            f"{MIN_DURATION_SECONDS:.0f}-{MAX_DURATION_SECONDS:.0f}s"
        )
    if problems:
        raise RecordingCollectionError(f"{source}: " + "; ".join(problems))
    return duration


def _prepare_source(path: Path) -> ProcessedAudio:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise RecordingCollectionError(f"could not inspect source recording: {path}") from error
    if size <= 0:
        raise RecordingCollectionError(f"source recording is empty: {path}")
    if size > MAX_SOURCE_BYTES:
        raise RecordingCollectionError(f"source recording exceeds 25 MB: {path} ({size} bytes)")
    try:
        decoded = decode_audio(path)
    except AudioDecodingError as error:
        raise RecordingCollectionError(f"source recording could not be decoded: {path}") from error
    if decoded.frames <= 0:
        raise RecordingCollectionError(f"source recording contains no frames: {path}")
    if not np.isfinite(decoded.samples).all():
        raise RecordingCollectionError(f"source recording contains non-finite samples: {path}")
    if not MIN_DURATION_SECONDS <= decoded.duration_seconds <= MAX_DURATION_SECONDS:
        raise RecordingCollectionError(
            f"source recording duration must be 5-20 seconds: {path} "
            f"({decoded.duration_seconds:.3f}s)"
        )
    try:
        processed = preprocess_audio(decoded, TARGET_SAMPLE_RATE)
    except (AudioValidationError, ValueError) as error:
        raise RecordingCollectionError(
            f"source recording could not be normalized: {path}"
        ) from error
    if not MIN_DURATION_SECONDS <= processed.duration_seconds <= MAX_DURATION_SECONDS:
        raise RecordingCollectionError(
            f"normalized recording duration must be 5-20 seconds: {path} "
            f"({processed.duration_seconds:.3f}s)"
        )
    return processed


def _write_canonical_audio(audio: ProcessedAudio, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise RecordingCollectionError(f"refusing to overwrite audio: {destination}")
    temporary = NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".tmp.wav", dir=destination.parent, delete=False
    )
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        sf.write(
            temporary_path,
            audio.samples,
            audio.sample_rate,
            format="WAV",
            subtype="PCM_16",
        )
        inspect_canonical_audio(temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists() and not overwrite:
            raise RecordingCollectionError(f"refusing to overwrite audio: {destination}")
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def discover_recordings(raw_directory: str | Path) -> dict[str, Path]:
    """Find one supported, non-recursive source file per recording ID."""
    directory = Path(raw_directory)
    if not directory.is_dir():
        raise RecordingCollectionError(f"raw recording directory does not exist: {directory}")
    candidates: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            continue
        recording_id = path.stem
        if recording_id in candidates:
            raise RecordingCollectionError(
                f"multiple source files found for {recording_id}: "
                f"{candidates[recording_id].name}, {path.name}"
            )
        candidates[recording_id] = path
    if not candidates:
        raise RecordingCollectionError(
            f"no supported recordings found in {directory}; expected names like "
            "speaker-01-prompt-01.wav"
        )
    return candidates


def audit_collection(
    dataset: EvaluationDataset, checklist: Mapping[str, ChecklistEntry]
) -> CollectionProgress:
    """Report real file/checklist progress without reading or emitting transcripts."""
    consented = 0
    audio_present = 0
    valid_audio = 0
    recorded = 0
    transcript_reviewed = 0
    spans_reviewed = 0
    ready = 0
    issues: list[str] = []
    for sample in dataset.samples:
        entry = checklist[sample.id]
        consented += int(entry.consent_confirmed)
        recorded += int(entry.recorded)
        transcript_reviewed += int(entry.transcript_reviewed)
        spans_reviewed += int(entry.entity_spans_reviewed)
        present = sample.file.is_file()
        audio_present += int(present)
        valid = False
        if present:
            try:
                inspect_canonical_audio(sample.file)
                valid = True
                valid_audio += 1
            except RecordingCollectionError as error:
                issues.append(f"{sample.id}: {error}")
        if entry.recorded and not present:
            issues.append(f"{sample.id}: checklist says recorded but audio is missing")
        if present and not entry.recorded:
            issues.append(f"{sample.id}: audio exists but checklist recorded is false")
        if entry.recorded and not entry.consent_confirmed:
            issues.append(f"{sample.id}: recorded is true without confirmed consent")
        if entry.transcript_reviewed and not entry.recorded:
            issues.append(f"{sample.id}: transcript review is true before recording")
        if entry.entity_spans_reviewed and not entry.transcript_reviewed:
            issues.append(f"{sample.id}: entity-span review is true before transcript review")
        if (
            entry.consent_confirmed
            and entry.recorded
            and entry.transcript_reviewed
            and entry.entity_spans_reviewed
            and valid
        ):
            ready += 1

    planned = len(dataset.samples)
    status = (
        "ready_for_evaluation"
        if ready == planned and not issues
        else "inconsistent"
        if issues
        else "collection_in_progress"
    )
    return CollectionProgress(
        status=status,
        planned_count=planned,
        consented_count=consented,
        audio_present_count=audio_present,
        valid_audio_count=valid_audio,
        recorded_checklist_count=recorded,
        transcript_reviewed_count=transcript_reviewed,
        entity_spans_reviewed_count=spans_reviewed,
        ready_for_evaluation_count=ready,
        remaining_count=planned - ready,
        issues=tuple(issues),
    )


def import_recordings(
    *,
    metadata_path: str | Path,
    checklist_path: str | Path,
    raw_directory: str | Path,
    overwrite: bool = False,
) -> ImportResult:
    """Import exact-ID recordings only after checklist consent is explicitly true."""
    dataset = load_dataset(metadata_path, require_files=False)
    checklist = load_checklist(checklist_path, dataset)
    candidates = discover_recordings(raw_directory)
    samples = {sample.id: sample for sample in dataset.samples}
    unexpected = sorted(set(candidates) - set(samples))
    if unexpected:
        raise RecordingCollectionError(f"source recording IDs are not planned: {unexpected}")

    already_present: list[str] = []
    to_write: list[tuple[str, Path, Path, bool]] = []
    for recording_id, source in candidates.items():
        entry = checklist[recording_id]
        sample = samples[recording_id]
        if not entry.consent_confirmed:
            raise RecordingCollectionError(
                f"{recording_id}: set consent_confirmed=true only after explicit consent"
            )
        if source.resolve() == sample.file.resolve():
            raise RecordingCollectionError(
                f"{recording_id}: raw source and evaluation destination must differ"
            )
        destination_exists = sample.file.exists()
        if destination_exists and not overwrite:
            inspect_canonical_audio(sample.file)
            already_present.append(recording_id)
            continue
        _prepare_source(source)
        to_write.append((recording_id, source, sample.file, destination_exists))

    imported: list[str] = []
    replaced: list[str] = []
    for recording_id, source, destination, destination_existed in to_write:
        processed = _prepare_source(source)
        _write_canonical_audio(processed, destination, overwrite=overwrite)
        (replaced if destination_existed else imported).append(recording_id)

    updated = dict(checklist)
    for recording_id in candidates:
        updated[recording_id] = replace(updated[recording_id], recorded=True)
    write_checklist(checklist_path, updated)
    progress = audit_collection(dataset, updated)
    return ImportResult(
        imported_count=len(imported),
        replaced_count=len(replaced),
        already_present_count=len(already_present),
        recording_ids=tuple(sorted(candidates)),
        progress=progress,
    )


def _add_collection_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--metadata", default="data/private_test/metadata.jsonl")
    parser.add_argument("--checklist", default="data/private_test/recording_checklist.csv")
    parser.add_argument("--progress", default="reports/evaluation_collection_progress.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="audit current recording progress")
    _add_collection_paths(audit_parser)
    import_parser = commands.add_parser(
        "import", help="import consented recordings from one directory"
    )
    _add_collection_paths(import_parser)
    import_parser.add_argument("--raw-directory", required=True)
    import_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_args(argv)
    metadata_path = resolve_project_path(arguments.metadata)
    checklist_path = resolve_project_path(arguments.checklist)
    progress_path = resolve_project_path(arguments.progress)
    try:
        if arguments.command == "import":
            result = import_recordings(
                metadata_path=metadata_path,
                checklist_path=checklist_path,
                raw_directory=resolve_project_path(arguments.raw_directory),
                overwrite=bool(arguments.overwrite),
            )
            progress = result.progress
            LOGGER.info(
                "Imported %s, replaced %s, already present %s; source files were kept.",
                result.imported_count,
                result.replaced_count,
                result.already_present_count,
            )
        else:
            dataset = load_dataset(metadata_path, require_files=False)
            checklist = load_checklist(checklist_path, dataset)
            progress = audit_collection(dataset, checklist)
        write_json_atomic(progress.to_dict(), progress_path)
    except (DatasetValidationError, OSError, RecordingCollectionError) as error:
        LOGGER.error("recording collection failed: %s", error)
        return 2
    LOGGER.info(
        "Collection status: %s (%s/%s ready).",
        progress.status,
        progress.ready_for_evaluation_count,
        progress.planned_count,
    )
    return 1 if progress.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
