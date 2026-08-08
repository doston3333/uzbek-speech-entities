"""Generate a non-destructive, consent-first 150-recording Phase 8 manifest."""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.dataset import (
    APPLICATION_LABELS,
    REQUIRED_CONDITIONS,
    REQUIRED_CONTENT_COVERAGE,
    GoldEntity,
)
from evaluation.io_utils import write_csv_atomic, write_json_atomic, write_jsonl_atomic
from uzbek_speech_entities.config import resolve_project_path

LOGGER = logging.getLogger(__name__)
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_PROMPT_KEYS = frozenset(
    {"id", "text", "entities", "speech_conditions", "content_tags"}
)
_PROMPT_ENTITY_KEYS = frozenset({"text", "label"})
_SPEAKER_KEYS = frozenset({"speaker_id", "conditions"})


class RecordingPlanError(ValueError):
    """Raised when a collection plan cannot prove its intended coverage."""


@dataclass(frozen=True, slots=True)
class PromptEntity:
    text: str
    label: str


@dataclass(frozen=True, slots=True)
class RecordingPrompt:
    id: str
    text: str
    entities: tuple[PromptEntity, ...]
    speech_conditions: tuple[str, ...]
    content_tags: tuple[str, ...]

    def gold_entities(self) -> tuple[GoldEntity, ...]:
        """Resolve unique prompt surfaces to deterministic character spans."""
        resolved: list[GoldEntity] = []
        for entity in self.entities:
            start = self.text.find(entity.text)
            if start < 0 or self.text.find(entity.text, start + 1) >= 0:
                raise RecordingPlanError(
                    f"{self.id}: entity surface must occur exactly once: {entity.text}"
                )
            resolved.append(
                GoldEntity(
                    text=entity.text,
                    label=entity.label,
                    start=start,
                    end=start + len(entity.text),
                )
            )
        resolved.sort(key=lambda entity: (entity.start, entity.end))
        for previous, current in zip(resolved, resolved[1:], strict=False):
            if current.start < previous.end:
                raise RecordingPlanError(f"{self.id}: prompt entities overlap")
        return tuple(resolved)


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    speaker_id: str
    conditions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecordingPlan:
    metadata_rows: tuple[Mapping[str, object], ...]
    checklist_rows: tuple[Mapping[str, object], ...]
    summary: Mapping[str, object]


def _mapping(value: object, expected: frozenset[str], location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RecordingPlanError(f"{location} must be an object")
    keys = frozenset(value)
    if keys != expected:
        raise RecordingPlanError(
            f"{location} schema mismatch; missing={sorted(expected - keys)}, "
            f"unexpected={sorted(keys - expected)}"
        )
    return value


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecordingPlanError(f"{location} must be non-empty text")
    return value


def _identifier(value: object, location: str) -> str:
    identifier = _text(value, location)
    if _SAFE_ID.fullmatch(identifier) is None:
        raise RecordingPlanError(f"{location} must use lowercase letters, digits, and hyphens")
    return identifier


def _strings(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RecordingPlanError(f"{location} must be a non-empty array")
    items = tuple(_text(item, f"{location}[]") for item in value)
    if len(set(items)) != len(items):
        raise RecordingPlanError(f"{location} must not contain duplicates")
    return items


def load_prompts(path: str | Path) -> tuple[RecordingPrompt, ...]:
    """Load strict JSONL prompts and validate all entity surfaces before recording."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RecordingPlanError(f"could not read prompt pack: {source}") from error
    if not lines or any(not line.strip() for line in lines):
        raise RecordingPlanError("prompt pack must contain non-blank JSONL records")
    prompts: list[RecordingPrompt] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            raw: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecordingPlanError(f"invalid prompt JSON at line {line_number}") from error
        values = _mapping(raw, _PROMPT_KEYS, f"prompt line {line_number}")
        prompt_id = _identifier(values["id"], f"prompt line {line_number}.id")
        if prompt_id in seen_ids:
            raise RecordingPlanError(f"duplicate prompt id: {prompt_id}")
        prompt_text = _text(values["text"], f"prompt line {line_number}.text")
        word_count = len(prompt_text.split())
        if not 12 <= word_count <= 38:
            raise RecordingPlanError(
                f"{prompt_id}: expected 12-38 words for a 5-20 second recording; found "
                f"{word_count}"
            )
        raw_entities = values["entities"]
        if not isinstance(raw_entities, list):
            raise RecordingPlanError(f"{prompt_id}.entities must be an array")
        entities: list[PromptEntity] = []
        for entity_index, raw_entity in enumerate(raw_entities):
            entity = _mapping(
                raw_entity, _PROMPT_ENTITY_KEYS, f"{prompt_id}.entities[{entity_index}]"
            )
            label = _text(entity["label"], f"{prompt_id}.entities[{entity_index}].label")
            if label not in APPLICATION_LABELS:
                raise RecordingPlanError(
                    f"{prompt_id}.entities[{entity_index}].label must be public"
                )
            entities.append(
                PromptEntity(
                    text=_text(
                        entity["text"], f"{prompt_id}.entities[{entity_index}].text"
                    ),
                    label=label,
                )
            )
        if {entity.label for entity in entities} != set(APPLICATION_LABELS):
            raise RecordingPlanError(f"{prompt_id} must cover PER, LOC, ORG, and DATE")
        speech_conditions = _strings(
            values["speech_conditions"], f"{prompt_id}.speech_conditions"
        )
        if not set(speech_conditions) <= REQUIRED_CONDITIONS:
            raise RecordingPlanError(f"{prompt_id} has unknown speech conditions")
        content_tags = _strings(values["content_tags"], f"{prompt_id}.content_tags")
        if not set(content_tags) <= REQUIRED_CONTENT_COVERAGE:
            raise RecordingPlanError(f"{prompt_id} has unknown content tags")
        prompt = RecordingPrompt(
            id=prompt_id,
            text=prompt_text,
            entities=tuple(entities),
            speech_conditions=speech_conditions,
            content_tags=content_tags,
        )
        prompt.gold_entities()
        prompts.append(prompt)
        seen_ids.add(prompt_id)
    return tuple(prompts)


def load_speaker_profiles(path: str | Path) -> tuple[SpeakerProfile, ...]:
    source = Path(path)
    try:
        raw: object = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RecordingPlanError(f"could not read speaker plan: {source}") from error
    if not isinstance(raw, list) or not raw:
        raise RecordingPlanError("speaker plan must be a non-empty JSON array")
    profiles: list[SpeakerProfile] = []
    seen_ids: set[str] = set()
    for index, raw_profile in enumerate(raw):
        values = _mapping(raw_profile, _SPEAKER_KEYS, f"speaker[{index}]")
        speaker_id = _identifier(values["speaker_id"], f"speaker[{index}].speaker_id")
        if speaker_id in seen_ids:
            raise RecordingPlanError(f"duplicate speaker id: {speaker_id}")
        conditions = _strings(values["conditions"], f"speaker[{index}].conditions")
        if not set(conditions) <= REQUIRED_CONDITIONS:
            raise RecordingPlanError(f"{speaker_id} has unknown recording conditions")
        profiles.append(SpeakerProfile(speaker_id=speaker_id, conditions=conditions))
        seen_ids.add(speaker_id)
    if len(profiles) < 5:
        raise RecordingPlanError(
            f"at least five speaker profiles are required; found {len(profiles)}"
        )
    return tuple(profiles)


def build_recording_plan(
    prompts: Sequence[RecordingPrompt], speakers: Sequence[SpeakerProfile]
) -> RecordingPlan:
    """Cross prompts and consented speaker slots, then prove planned coverage."""
    if not prompts or not speakers:
        raise RecordingPlanError("prompts and speakers must not be empty")
    metadata_rows: list[Mapping[str, object]] = []
    checklist_rows: list[Mapping[str, object]] = []
    entity_counts: Counter[str] = Counter()
    coverage: set[str] = set()
    for speaker in speakers:
        for prompt in prompts:
            recording_id = f"{speaker.speaker_id}-{prompt.id}"
            conditions = tuple(
                sorted(
                    set(speaker.conditions)
                    | set(prompt.speech_conditions)
                    | set(prompt.content_tags)
                )
            )
            gold_entities = prompt.gold_entities()
            entity_counts.update(entity.label for entity in gold_entities)
            coverage.update(conditions)
            relative_file = f"audio/{recording_id}.wav"
            metadata_rows.append(
                {
                    "id": recording_id,
                    "file": relative_file,
                    "gold_transcript": prompt.text,
                    "entities": [
                        {
                            "text": entity.text,
                            "label": entity.label,
                            "start": entity.start,
                            "end": entity.end,
                        }
                        for entity in gold_entities
                    ],
                    "speaker_id": speaker.speaker_id,
                    "conditions": list(conditions),
                }
            )
            checklist_rows.append(
                {
                    "recording_id": recording_id,
                    "speaker_id": speaker.speaker_id,
                    "prompt_id": prompt.id,
                    "audio_file": relative_file,
                    "consent_confirmed": False,
                    "recorded": False,
                    "transcript_reviewed": False,
                    "entity_spans_reviewed": False,
                }
            )
    recording_count = len(metadata_rows)
    if not 100 <= recording_count <= 300:
        raise RecordingPlanError(
            f"planned recording count must be 100-300; found {recording_count}"
        )
    for label in APPLICATION_LABELS:
        if entity_counts[label] < 30:
            raise RecordingPlanError(
                f"planned {label} mentions must be at least 30; found {entity_counts[label]}"
            )
    missing_conditions = REQUIRED_CONDITIONS - coverage
    missing_content = REQUIRED_CONTENT_COVERAGE - coverage
    if missing_conditions or missing_content:
        raise RecordingPlanError(
            "planned coverage is incomplete; "
            f"conditions={sorted(missing_conditions)}, content={sorted(missing_content)}"
        )
    summary: Mapping[str, object] = {
        "status": "planned_not_recorded",
        "recorded_count": 0,
        "planned_recording_count": recording_count,
        "prompt_count": len(prompts),
        "speaker_count": len(speakers),
        "entity_counts": {label: entity_counts[label] for label in APPLICATION_LABELS},
        "condition_coverage": sorted(coverage & REQUIRED_CONDITIONS),
        "content_coverage": sorted(coverage & REQUIRED_CONTENT_COVERAGE),
        "consent_requirement": (
            "Every speaker must explicitly consent before their checklist rows are recorded."
        ),
        "limitations": [
            "The plan is not evaluation evidence until every referenced audio file exists.",
            "Actual duration, microphone, noise, speaking rate, and annotation quality must be "
            "validated after recording.",
        ],
    }
    return RecordingPlan(tuple(metadata_rows), tuple(checklist_rows), summary)


def write_recording_plan(
    plan: RecordingPlan,
    *,
    metadata_path: str | Path,
    checklist_path: str | Path,
    summary_path: str | Path,
    overwrite: bool = False,
) -> None:
    outputs = tuple(Path(path) for path in (metadata_path, checklist_path, summary_path))
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing collection files: "
            + ", ".join(str(path) for path in existing)
        )
    write_jsonl_atomic(plan.metadata_rows, outputs[0])
    write_csv_atomic(
        plan.checklist_rows,
        (
            "recording_id",
            "speaker_id",
            "prompt_id",
            "audio_file",
            "consent_confirmed",
            "recorded",
            "transcript_reviewed",
            "entity_spans_reviewed",
        ),
        outputs[1],
    )
    write_json_atomic(plan.summary, outputs[2])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="data/private_test/prompts.jsonl")
    parser.add_argument("--speakers", default="data/private_test/speakers.example.json")
    parser.add_argument("--metadata", default="data/private_test/metadata.jsonl")
    parser.add_argument("--checklist", default="data/private_test/recording_checklist.csv")
    parser.add_argument("--summary", default="reports/evaluation_collection_plan.json")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = parse_args(argv)
    try:
        prompts = load_prompts(resolve_project_path(arguments.prompts))
        speakers = load_speaker_profiles(resolve_project_path(arguments.speakers))
        plan = build_recording_plan(prompts, speakers)
        write_recording_plan(
            plan,
            metadata_path=resolve_project_path(arguments.metadata),
            checklist_path=resolve_project_path(arguments.checklist),
            summary_path=resolve_project_path(arguments.summary),
            overwrite=bool(arguments.overwrite),
        )
    except (FileExistsError, OSError, RecordingPlanError, ValueError) as error:
        LOGGER.error("recording plan failed: %s", error)
        return 2
    LOGGER.info(
        "Planned %s recordings; no audio was created or claimed as complete.",
        plan.summary["planned_recording_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
