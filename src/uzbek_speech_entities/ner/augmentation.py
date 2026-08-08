"""Deterministic, label-safe augmentation for prepared NER training records.

The transformations run in the order declared by ``TRANSFORMATION_ORDER``.
They are deliberately limited to tokenization and surface-form variation: no
spelling noise or semantic substitution is introduced.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .labels import BIOValidationError, validate_bio_record, validate_bio_sequence

DEFAULT_SEED = 42
DEFAULT_AUGMENTATION_NUMERATOR = 3
DEFAULT_AUGMENTATION_DENOMINATOR = 7
TRANSFORMATION_ORDER = (
    "lowercase",
    "remove_punctuation",
    "apostrophe_representation",
    "reduce_capitalization",
    "hyphen_spacing",
    "colon_spacing",
)
_TRANSFORMATION_SET = frozenset(TRANSFORMATION_ORDER)
_APOSTROPHES = frozenset(("'", "’", "ʻ", "ʼ", "‘", "`"))
_PUNCTUATION_CHARACTERS = frozenset((".", ",", "!", "?", ";", "…"))


def _lowercase(tokens: list[str], ner_tags: list[str]) -> tuple[list[str], list[str]]:
    return [token.lower() for token in tokens], ner_tags


def _remove_punctuation(tokens: list[str], ner_tags: list[str]) -> tuple[list[str], list[str]]:
    start = 0
    end = len(tokens)
    while (
        start < end
        and ner_tags[start] == "O"
        and all(character in _PUNCTUATION_CHARACTERS for character in tokens[start])
    ):
        start += 1
    while (
        end > start
        and ner_tags[end - 1] == "O"
        and all(character in _PUNCTUATION_CHARACTERS for character in tokens[end - 1])
    ):
        end -= 1
    if start == end:
        return tokens, ner_tags
    return tokens[start:end], ner_tags[start:end]


def _apostrophe_representation(
    tokens: list[str], ner_tags: list[str]
) -> tuple[list[str], list[str]]:
    converted_tokens: list[str] = []
    for token in tokens:
        converted = list(token)
        for index, character in enumerate(converted):
            is_internal = (
                0 < index < len(converted) - 1
                and converted[index - 1].isalnum()
                and converted[index + 1].isalnum()
            )
            if is_internal and character in _APOSTROPHES:
                converted[index] = "ʻ" if character == "'" else "'"
        converted_tokens.append("".join(converted))
    return converted_tokens, ner_tags


def _reduce_capitalization(
    tokens: list[str], ner_tags: list[str]
) -> tuple[list[str], list[str]]:
    reduced_tokens = [
        token.lower() if tag != "O" else token
        for token, tag in zip(tokens, ner_tags, strict=True)
    ]
    return reduced_tokens, ner_tags


def _expanded_tags(tag: str, count: int) -> list[str]:
    if count == 1 or tag == "O" or tag.startswith("I-"):
        return [tag] * count
    entity_type = tag.split("-", maxsplit=1)[1]
    return [tag, *([f"I-{entity_type}"] * (count - 1))]


def _hyphen_spacing(tokens: list[str], ner_tags: list[str]) -> tuple[list[str], list[str]]:
    changed_tokens: list[str] = []
    changed_tags: list[str] = []
    for token, tag in zip(tokens, ner_tags, strict=True):
        if token == "-" and tag == "O":
            continue
        split_indexes = [
            index
            for index, character in enumerate(token)
            if (
                character == "-"
                and any(part.isdigit() for part in token)
                and 0 < index < len(token) - 1
                and token[index - 1].isalnum()
                and token[index + 1].isalnum()
            )
        ]
        if not split_indexes:
            changed_tokens.append(token)
            changed_tags.append(tag)
            continue
        parts: list[str] = []
        start = 0
        for index in split_indexes:
            parts.append(token[start:index])
            start = index + 1
        parts.append(token[start:])
        changed_tokens.extend(parts)
        changed_tags.extend(_expanded_tags(tag, len(parts)))
    return changed_tokens, changed_tags


def _colon_spacing(tokens: list[str], ner_tags: list[str]) -> tuple[list[str], list[str]]:
    changed_tokens: list[str] = []
    changed_tags: list[str] = []
    for token, tag in zip(tokens, ner_tags, strict=True):
        split_indexes = [
            index
            for index, character in enumerate(token)
            if (
                character == ":"
                and 0 < index < len(token) - 1
                and token[index - 1].isdigit()
                and token[index + 1].isdigit()
            )
        ]
        if not split_indexes:
            changed_tokens.append(token)
            changed_tags.append(tag)
            continue
        parts: list[str] = []
        start = 0
        for index in split_indexes:
            parts.extend((token[start:index], ":"))
            start = index + 1
        parts.append(token[start:])
        changed_tokens.extend(parts)
        changed_tags.extend(_expanded_tags(tag, len(parts)))
    return changed_tokens, changed_tags


_TRANSFORMS = {
    "lowercase": _lowercase,
    "remove_punctuation": _remove_punctuation,
    "apostrophe_representation": _apostrophe_representation,
    "reduce_capitalization": _reduce_capitalization,
    "hyphen_spacing": _hyphen_spacing,
    "colon_spacing": _colon_spacing,
}


def apply_transformations(
    tokens: Sequence[str], ner_tags: Sequence[str], transformations: Iterable[str]
) -> tuple[list[str], list[str], tuple[str, ...]]:
    """Apply requested transformations in fixed order and return names that changed data."""
    validate_bio_sequence(tokens, ner_tags)
    requested = frozenset(transformations)
    unknown = sorted(requested - _TRANSFORMATION_SET)
    if unknown:
        raise ValueError(f"unknown augmentation transformation(s): {unknown!r}")
    if {"lowercase", "reduce_capitalization"} <= requested:
        raise ValueError("lowercase and reduce_capitalization are mutually exclusive")

    current_tokens = list(tokens)
    current_tags = list(ner_tags)
    applied: list[str] = []
    for name in TRANSFORMATION_ORDER:
        if name not in requested:
            continue
        transformed_tokens, transformed_tags = _TRANSFORMS[name](current_tokens, current_tags)
        if (transformed_tokens, transformed_tags) != (current_tokens, current_tags):
            applied.append(name)
        current_tokens, current_tags = transformed_tokens, transformed_tags
    validate_bio_sequence(current_tokens, current_tags)
    return current_tokens, current_tags, tuple(applied)


def _applicable_transformations(record: Mapping[str, object]) -> list[str]:
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    assert isinstance(tokens, Sequence) and not isinstance(tokens, str | bytes)
    assert isinstance(ner_tags, Sequence) and not isinstance(ner_tags, str | bytes)
    return [
        name
        for name in TRANSFORMATION_ORDER
        if apply_transformations(tokens, ner_tags, (name,))[2]
    ]


def _choose_transformations(applicable: Sequence[str], rng: random.Random) -> frozenset[str]:
    capitalization = [
        name for name in ("lowercase", "reduce_capitalization") if name in applicable
    ]
    others = [name for name in applicable if name not in capitalization]
    candidates = others + ([rng.choice(capitalization)] if capitalization else [])
    rng.shuffle(candidates)
    count = rng.randint(1, min(3, len(candidates)))
    return frozenset(candidates[:count])


def _derived_id(source_id: str, ordinal: int, existing_ids: set[str]) -> str:
    candidate = f"{source_id}__aug_{ordinal:04d}"
    suffix = 2
    while candidate in existing_ids:
        candidate = f"{source_id}__aug_{ordinal:04d}_{suffix}"
        suffix += 1
    return candidate


def _selected_source_digest(source_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()


def augment_records(
    records: Sequence[Mapping[str, object]], seed: int = DEFAULT_SEED
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append deterministic, safe variants at ``round(N * 3 / 7)`` of the input size."""
    source_records: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise BIOValidationError(f"source record at index {index} is not an object")
        source_record = dict(record)
        try:
            validate_bio_record(source_record)
        except BIOValidationError as error:
            raise BIOValidationError(f"invalid source record at index {index}: {error}") from error
        if source_record.get("augmentation") is not None:
            raise ValueError(
                f"source record at index {index} is already augmented and cannot be re-augmented"
            )
        source_id = source_record["id"]
        assert isinstance(source_id, str)
        if source_id in source_ids:
            raise ValueError(f"duplicate source record ID: {source_id!r}")
        source_ids.add(source_id)
        source_records.append(source_record)

    requested_count = round(
        len(source_records) * DEFAULT_AUGMENTATION_NUMERATOR / DEFAULT_AUGMENTATION_DENOMINATOR
    )
    viable = [
        record for record in source_records if _applicable_transformations(record)
    ]
    if requested_count > len(viable):
        raise ValueError(
            "not enough source records with applicable safe transformations: "
            f"need {requested_count}, found {len(viable)}"
        )

    rng = random.Random(seed)
    rng.shuffle(viable)
    selected = viable[:requested_count]
    output_records = [dict(record) for record in source_records]
    existing_ids = set(source_ids)
    transformation_counts: Counter[str] = Counter()
    selected_source_ids: list[str] = []

    for ordinal, source_record in enumerate(selected, start=1):
        source_id = source_record["id"]
        tokens = source_record["tokens"]
        ner_tags = source_record["ner_tags"]
        assert isinstance(source_id, str)
        assert isinstance(tokens, Sequence) and not isinstance(tokens, str | bytes)
        assert isinstance(ner_tags, Sequence) and not isinstance(ner_tags, str | bytes)
        requested = _choose_transformations(_applicable_transformations(source_record), rng)
        transformed_tokens, transformed_tags, applied = apply_transformations(
            tokens, ner_tags, requested
        )
        if not applied or (transformed_tokens, transformed_tags) == (list(tokens), list(ner_tags)):
            raise RuntimeError(f"selected augmentation did not change source record {source_id!r}")
        augmented_id = _derived_id(source_id, ordinal, existing_ids)
        augmented_record = dict(source_record)
        augmented_record["id"] = augmented_id
        augmented_record["tokens"] = transformed_tokens
        augmented_record["ner_tags"] = transformed_tags
        augmented_record["augmentation"] = {
            "source_id": source_id,
            "transformations": list(applied),
        }
        validate_bio_record(augmented_record)
        existing_ids.add(augmented_id)
        output_records.append(augmented_record)
        selected_source_ids.append(source_id)
        transformation_counts.update(applied)

    statistics: dict[str, Any] = {
        "allowed_transformations": list(TRANSFORMATION_ORDER),
        "augmented_record_count": len(selected),
        "augmentation_fraction_of_output": round(len(selected) / len(output_records), 6)
        if output_records
        else 0.0,
        "augmentation_to_source_ratio": round(len(selected) / len(source_records), 6)
        if source_records
        else 0.0,
        "output_record_count": len(output_records),
        "requested_augmented_record_count": requested_count,
        "seed": seed,
        "selected_source_ids_sha256": _selected_source_digest(selected_source_ids),
        "selected_source_record_count": len(selected_source_ids),
        "source_record_count": len(source_records),
        "transformation_counts": {
            name: transformation_counts[name] for name in TRANSFORMATION_ORDER
        },
    }
    return output_records, statistics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} is not an object")
            records.append(record)
    return records


def _atomic_write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary_file:
            temporary_name = temporary_file.name
            for record in records:
                temporary_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                temporary_file.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary_file:
            temporary_name = temporary_file.name
            json.dump(value, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def augment_training_file(
    input_path: Path,
    output_path: Path,
    statistics_path: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Augment one training JSONL file and atomically write its deterministic artifacts."""
    protected_names = {"validation.jsonl", "test.jsonl"}
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if resolved_input == resolved_output:
        raise ValueError("augmentation input and output paths must be different")
    if input_path.name in protected_names or output_path.name in protected_names:
        raise ValueError("augmentation is restricted to training data, never validation or test")
    if statistics_path is not None:
        resolved_statistics = statistics_path.resolve()
        if resolved_statistics in {resolved_input, resolved_output}:
            raise ValueError("augmentation statistics path must differ from input and output paths")
        if statistics_path.name in protected_names:
            raise ValueError("augmentation statistics path must not target validation or test data")
    if not input_path.is_file():
        raise FileNotFoundError(f"missing source training JSONL file: {input_path}")
    records, statistics = augment_records(_read_jsonl(input_path), seed=seed)
    statistics = {
        **statistics,
        "input_path": str(input_path),
        "output_path": str(output_path),
    }
    _atomic_write_jsonl(output_path, records)
    if statistics_path is not None:
        _atomic_write_json(statistics_path, statistics)
    return statistics
