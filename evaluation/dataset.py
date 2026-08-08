"""Strict loading and compliance reporting for private evaluation metadata."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import soundfile as sf  # type: ignore[import-untyped]

APPLICATION_LABELS: Final[tuple[str, ...]] = ("PER", "LOC", "ORG", "DATE")
REQUIRED_CONDITIONS: Final[frozenset[str]] = frozenset(
    {
        "quiet",
        "background_noise",
        "laptop_microphone",
        "phone_microphone",
        "fast_speech",
        "slow_speech",
        "formal_speech",
        "conversational_speech",
        "uzbek_russian_code_switching",
    }
)
REQUIRED_CONTENT_COVERAGE: Final[frozenset[str]] = frozenset(
    {
        "common_names",
        "rare_names",
        "cities",
        "regions",
        "districts",
        "villages",
        "organizations",
        "relative_dates",
        "numeric_dates",
        "times",
        "inflected_entities",
    }
)
_SAMPLE_KEYS: Final[frozenset[str]] = frozenset(
    {"id", "file", "gold_transcript", "entities", "speaker_id", "conditions"}
)
_ENTITY_KEYS: Final[frozenset[str]] = frozenset({"text", "label", "start", "end"})


class DatasetValidationError(ValueError):
    """Raised when metadata does not follow the private evaluation schema."""


@dataclass(frozen=True, slots=True)
class GoldEntity:
    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    id: str
    file: Path
    gold_transcript: str
    entities: tuple[GoldEntity, ...]
    speaker_id: str
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    metadata_path: Path
    samples: tuple[EvaluationSample, ...]


@dataclass(frozen=True, slots=True)
class AudioInspection:
    file: str
    duration_seconds: float | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    recording_count: int
    speaker_count: int
    entity_counts: Mapping[str, int]
    condition_coverage: tuple[str, ...]
    content_coverage: tuple[str, ...]
    audio: tuple[AudioInspection, ...]
    issues: tuple[str, ...]

    @property
    def compliant(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, object]:
        return {
            "recording_count": self.recording_count,
            "speaker_count": self.speaker_count,
            "entity_counts": dict(self.entity_counts),
            "condition_coverage": list(self.condition_coverage),
            "content_coverage": list(self.content_coverage),
            "audio": [asdict(item) for item in self.audio],
            "issues": list(self.issues),
            "compliant": self.compliant,
        }


def _nonempty_string(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location} must be a non-empty string")
    return value


def _strict_mapping(value: object, expected: frozenset[str], location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DatasetValidationError(f"{location} must be an object")
    keys = frozenset(value)
    if keys != expected:
        missing = sorted(expected - keys)
        unexpected = sorted(keys - expected)
        raise DatasetValidationError(
            f"{location} schema mismatch; missing={missing}, unexpected={unexpected}"
        )
    return value


def _resolve_audio_path(raw_file: str, parent: Path, location: str) -> Path:
    candidate = Path(raw_file)
    if candidate.is_absolute():
        raise DatasetValidationError(f"{location} must be relative to the metadata file")
    resolved_parent = parent.resolve()
    resolved = (resolved_parent / candidate).resolve()
    try:
        resolved.relative_to(resolved_parent)
    except ValueError as exc:
        raise DatasetValidationError(f"{location} escapes the metadata directory") from exc
    return resolved


def _parse_entity(value: object, transcript: str, location: str) -> GoldEntity:
    data = _strict_mapping(value, _ENTITY_KEYS, location)
    text = _nonempty_string(data["text"], f"{location}.text")
    label = _nonempty_string(data["label"], f"{location}.label")
    if label not in APPLICATION_LABELS:
        raise DatasetValidationError(f"{location}.label must be one of {APPLICATION_LABELS}")
    start, end = data["start"], data["end"]
    if isinstance(start, bool) or not isinstance(start, int):
        raise DatasetValidationError(f"{location}.start must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise DatasetValidationError(f"{location}.end must be an integer")
    if start < 0 or end <= start or end > len(transcript):
        raise DatasetValidationError(f"{location} has invalid span [{start}, {end})")
    if transcript[start:end] != text:
        raise DatasetValidationError(f"{location}.text does not match gold_transcript span")
    return GoldEntity(text=text, label=label, start=start, end=end)


def _parse_conditions(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetValidationError(f"{location} must be a non-empty array")
    conditions = tuple(
        _nonempty_string(item, f"{location}[{index}]") for index, item in enumerate(value)
    )
    if len(set(conditions)) != len(conditions):
        raise DatasetValidationError(f"{location} must not contain duplicate values")
    return conditions


def load_dataset(metadata_path: str | Path, *, require_files: bool = False) -> EvaluationDataset:
    """Load exact-schema JSONL metadata without permitting unsafe audio paths."""
    path = Path(metadata_path).resolve()
    if not path.is_file():
        raise DatasetValidationError(f"metadata file does not exist: {path}")
    samples: list[EvaluationSample] = []
    ids: set[str] = set()
    files: set[Path] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetValidationError(f"could not read metadata file: {path}") from exc
    if not lines:
        raise DatasetValidationError("metadata file must contain at least one JSONL record")
    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise DatasetValidationError(f"line {line_number} must not be blank")
        try:
            raw_sample: object = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"line {line_number} is not valid JSON") from exc
        data = _strict_mapping(raw_sample, _SAMPLE_KEYS, f"line {line_number}")
        sample_id = _nonempty_string(data["id"], f"line {line_number}.id")
        if sample_id in ids:
            raise DatasetValidationError(f"duplicate id: {sample_id}")
        transcript = _nonempty_string(
            data["gold_transcript"], f"line {line_number}.gold_transcript"
        )
        speaker_id = _nonempty_string(data["speaker_id"], f"line {line_number}.speaker_id")
        raw_file = _nonempty_string(data["file"], f"line {line_number}.file")
        audio_file = _resolve_audio_path(raw_file, path.parent, f"line {line_number}.file")
        if audio_file in files:
            raise DatasetValidationError(f"duplicate file: {raw_file}")
        if require_files and not audio_file.is_file():
            raise DatasetValidationError(f"audio file does not exist: {audio_file}")
        raw_entities = data["entities"]
        if not isinstance(raw_entities, list):
            raise DatasetValidationError(f"line {line_number}.entities must be an array")
        entities = tuple(
            _parse_entity(item, transcript, f"line {line_number}.entities[{index}]")
            for index, item in enumerate(raw_entities)
        )
        for previous, current in zip(entities, entities[1:], strict=False):
            if current.start < previous.end:
                raise DatasetValidationError(
                    f"line {line_number}.entities must be ordered and non-overlapping"
                )
        samples.append(
            EvaluationSample(
                id=sample_id,
                file=audio_file,
                gold_transcript=transcript,
                entities=entities,
                speaker_id=speaker_id,
                conditions=_parse_conditions(data["conditions"], f"line {line_number}.conditions"),
            )
        )
        ids.add(sample_id)
        files.add(audio_file)
    return EvaluationDataset(metadata_path=path, samples=tuple(samples))


def inspect_audio(
    sample: EvaluationSample, *, display_path: str | None = None
) -> AudioInspection:
    """Read audio metadata only; errors are captured for honest compliance reports."""
    try:
        info = sf.info(sample.file)
        if info.samplerate <= 0:
            raise RuntimeError("audio sample rate must be positive")
        return AudioInspection(
            file=display_path or str(sample.file),
            duration_seconds=info.frames / info.samplerate,
            error=None,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as exc:
        return AudioInspection(
            file=display_path or str(sample.file), duration_seconds=None, error=str(exc)
        )


def build_compliance_report(
    dataset: EvaluationDataset, *, inspect_files: bool = True
) -> ComplianceReport:
    """Report Phase 8 requirements and all missing evidence as explicit issues."""
    counts = Counter(entity.label for sample in dataset.samples for entity in sample.entities)
    condition_values = frozenset(value for sample in dataset.samples for value in sample.conditions)
    audio = (
        tuple(
            inspect_audio(
                sample,
                display_path=str(sample.file.relative_to(dataset.metadata_path.parent)),
            )
            for sample in dataset.samples
        )
        if inspect_files
        else ()
    )
    issues: list[str] = []
    count = len(dataset.samples)
    if not 100 <= count <= 300:
        issues.append(f"recording count must be between 100 and 300; found {count}")
    speaker_count = len({sample.speaker_id for sample in dataset.samples})
    if speaker_count < 5:
        issues.append(f"at least 5 speakers are required; found {speaker_count}")
    for label in APPLICATION_LABELS:
        if counts[label] < 30:
            issues.append(f"at least 30 {label} mentions are required; found {counts[label]}")
    missing_conditions = REQUIRED_CONDITIONS - condition_values
    if missing_conditions:
        issues.append(f"missing required conditions: {', '.join(sorted(missing_conditions))}")
    missing_content = REQUIRED_CONTENT_COVERAGE - condition_values
    if missing_content:
        issues.append(f"missing required content coverage: {', '.join(sorted(missing_content))}")
    if not inspect_files:
        issues.append("audio durations were not inspected")
    for item in audio:
        if item.error is not None:
            issues.append(f"could not inspect audio {item.file}: {item.error}")
        elif item.duration_seconds is not None and not 5 <= item.duration_seconds <= 20:
            issues.append(
                "audio duration must be between 5 and 20 seconds: "
                f"{item.file} ({item.duration_seconds:.3f})"
            )
    return ComplianceReport(
        recording_count=count,
        speaker_count=speaker_count,
        entity_counts={label: counts[label] for label in APPLICATION_LABELS},
        condition_coverage=tuple(sorted(condition_values & REQUIRED_CONDITIONS)),
        content_coverage=tuple(sorted(condition_values & REQUIRED_CONTENT_COVERAGE)),
        audio=audio,
        issues=tuple(issues),
    )


def write_report(report: ComplianceReport, output_path: str | Path) -> None:
    """Atomically write a JSON compliance report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report.to_dict(), stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate private evaluation JSONL metadata")
    parser.add_argument("--metadata", required=True, help="Path to evaluation metadata JSONL")
    parser.add_argument("--output", required=True, help="Output compliance report JSON")
    parser.add_argument("--skip-audio-inspection", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        dataset = load_dataset(
            arguments.metadata, require_files=not arguments.skip_audio_inspection
        )
        report = build_compliance_report(dataset, inspect_files=not arguments.skip_audio_inspection)
        write_report(report, arguments.output)
    except DatasetValidationError as exc:
        parser.error(str(exc))
    return 0 if report.compliant else 1


if __name__ == "__main__":
    raise SystemExit(main())
