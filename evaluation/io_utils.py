"""Atomic, UTF-8 report writers shared by Phase 8 command-line tools."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _temporary_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    return descriptor, Path(name)


def write_json_atomic(value: object, output_path: str | Path) -> None:
    path = Path(output_path)
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_text_atomic(value: str, output_path: str | Path) -> None:
    path = Path(output_path)
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl_atomic(records: Iterable[Mapping[str, object]], output_path: str | Path) -> None:
    path = Path(output_path)
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read JSONL report: {source}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL record at {source}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL record at {source}:{line_number}") from error
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record must be an object at {source}:{line_number}")
        records.append(value)
    return records


def write_csv_atomic(
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
