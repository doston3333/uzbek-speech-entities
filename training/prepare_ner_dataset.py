"""Reconstruct, validate, split, and serialize the pinned Uzbek NER TSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from uzbek_speech_entities.ner.dataset import (
    DATASET_FILENAME,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SHA256,
    SOURCE_COLUMNS,
    sha256_file,
)
from uzbek_speech_entities.ner.labels import ENTITY_TYPES, BIOValidationError, validate_bio_record

REQUIRED_COLUMNS = SOURCE_COLUMNS
DEFAULT_INPUT = Path("data/raw") / DATASET_FILENAME
DEFAULT_OUTPUT_DIR = Path("data/processed/ner")
SPLIT_SEED = 42


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
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


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary_file:
            temporary_name = temporary_file.name
            json.dump(value, temporary_file, ensure_ascii=False, sort_keys=True, indent=2)
            temporary_file.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _parse_positive_integer(value: str | None, field: str) -> int:
    if value is None or not value.strip():
        raise ValueError(f"missing {field}")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"invalid {field}: {value!r}") from error
    if parsed <= 0:
        raise ValueError(f"nonpositive {field}: {value!r}")
    return parsed


def _source_row_error(row: dict[str, str | None]) -> tuple[int, int, str, str, str] | None:
    """Parse a row, returning a source diagnostic instead of raising on bad input."""
    sentence_value = row.get("Sentence")
    try:
        sentence = _parse_positive_integer(sentence_value, "Sentence")
    except ValueError:
        return None
    try:
        token_order = _parse_positive_integer(row.get("TokenOrder"), "TokenOrder")
    except ValueError as error:
        return (sentence, 0, "", "", str(error))
    token = row.get("Token")
    if token is None or not token.strip():
        return (sentence, token_order, "", "", "missing Token")
    tag = row.get("NER_Tag")
    if tag is None or not tag.strip():
        return (sentence, token_order, token, "", "missing NER_Tag")
    pos = row.get("pos")
    if pos is None or not pos.strip():
        return (sentence, token_order, token, tag, "missing pos")
    return (sentence, token_order, token, tag, "")


def _record_id(sentence: int) -> str:
    return f"sentence-{sentence:06d}"


def _length_summary(records: Sequence[dict[str, Any]]) -> dict[str, int | float]:
    lengths = [len(record["tokens"]) for record in records]
    if not lengths:
        return {"min_tokens": 0, "max_tokens": 0, "mean_tokens": 0.0, "total_tokens": 0}
    return {
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths), 6),
        "total_tokens": sum(lengths),
    }


def _split_records(records: Sequence[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Shuffle IDs, floor train/validation, and give the remaining sentences to test."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    record_count = len(shuffled)
    train_end = math.floor(record_count * 0.8)
    validation_end = train_end + math.floor(record_count * 0.1)
    split_records = {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }
    for split in split_records.values():
        split.sort(key=lambda record: record["id"])
    return split_records


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    seed: int = SPLIT_SEED,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Prepare local TSV records and return the deterministic statistics object."""
    source_sha256 = sha256_file(input_path)
    if expected_sha256 is not None and source_sha256 != expected_sha256:
        raise ValueError(
            f"source checksum mismatch: expected {expected_sha256}, got {source_sha256}"
        )

    with input_path.open(encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file, delimiter="\t")
        source_columns = reader.fieldnames
        expected_column_set = set(REQUIRED_COLUMNS)
        if (
            source_columns is None
            or set(source_columns) != expected_column_set
            or len(source_columns) != len(REQUIRED_COLUMNS)
        ):
            raise ValueError(
                "unexpected TSV columns: "
                f"{source_columns!r}; expected exactly {list(REQUIRED_COLUMNS)!r}"
            )

        grouped_rows: dict[int, list[tuple[int, str, str, int]]] = defaultdict(list)
        corrupt_sentences: dict[int, list[dict[str, Any]]] = defaultdict(list)
        rejected: list[dict[str, Any]] = []
        source_row_count = 0
        source_sentence_ids: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            source_row_count += 1
            parsed = _source_row_error(row)
            sentence_value = row.get("Sentence")
            try:
                sentence_for_diagnostic = _parse_positive_integer(sentence_value, "Sentence")
            except ValueError as error:
                rejected.append(
                    {
                        "kind": "source_row",
                        "reason": str(error),
                        "row": row,
                        "row_number": row_number,
                        "sentence": None,
                    }
                )
                continue
            source_sentence_ids.add(sentence_for_diagnostic)
            assert parsed is not None
            sentence, token_order, token, tag, error_message = parsed
            if error_message:
                diagnostic = {
                    "kind": "source_row",
                    "reason": error_message,
                    "row": row,
                    "row_number": row_number,
                    "sentence": sentence,
                }
                rejected.append(diagnostic)
                corrupt_sentences[sentence].append(diagnostic)
                continue
            grouped_rows[sentence].append((token_order, token, tag, row_number))

    accepted_records: list[dict[str, Any]] = []
    rejected_sentence_ids: set[int] = set(corrupt_sentences)
    for sentence in sorted(source_sentence_ids):
        if sentence in corrupt_sentences:
            rejected.append(
                {
                    "id": _record_id(sentence),
                    "kind": "sentence",
                    "reason": "contains invalid source row",
                    "row_numbers": sorted(
                        item["row_number"] for item in corrupt_sentences[sentence]
                    ),
                    "sentence": sentence,
                }
            )
            continue
        rows = grouped_rows.get(sentence, [])
        token_orders = [row[0] for row in rows]
        duplicate_orders = sorted(
            order for order, count in Counter(token_orders).items() if count > 1
        )
        if duplicate_orders:
            rejected_sentence_ids.add(sentence)
            rejected.append(
                {
                    "id": _record_id(sentence),
                    "kind": "sentence",
                    "reason": f"duplicate TokenOrder values: {duplicate_orders}",
                    "row_numbers": sorted(row[3] for row in rows),
                    "rows": [
                        {
                            "NER_Tag": row[2],
                            "Token": row[1],
                            "TokenOrder": row[0],
                            "row_number": row[3],
                        }
                        for row in rows
                    ],
                    "sentence": sentence,
                }
            )
            continue
        rows.sort(key=lambda row: row[0])
        record = {
            "id": _record_id(sentence),
            "tokens": [row[1] for row in rows],
            "ner_tags": [row[2] for row in rows],
            "source": DATASET_ID,
            "augmentation": None,
        }
        try:
            validate_bio_record(record)
        except BIOValidationError as error:
            rejected_sentence_ids.add(sentence)
            rejected.append(
                {
                    "id": _record_id(sentence),
                    "kind": "sentence",
                    "record": record,
                    "reason": str(error),
                    "row_numbers": [row[3] for row in rows],
                    "sentence": sentence,
                }
            )
            continue
        accepted_records.append(record)

    accepted_records.sort(key=lambda record: record["id"])
    splits = _split_records(accepted_records, seed)
    label_counts = Counter(tag for record in accepted_records for tag in record["ner_tags"])
    entity_type_counts = Counter(
        tag.split("-", maxsplit=1)[1] for tag in label_counts.elements() if tag != "O"
    )
    accepted_token_count = sum(len(record["tokens"]) for record in accepted_records)
    split_statistics = {
        name: {
            "actual_sentence_ratio": (
                round(len(split) / len(accepted_records), 6) if accepted_records else 0.0
            ),
            "actual_token_ratio": round(
                sum(len(record["tokens"]) for record in split) / accepted_token_count, 6
            )
            if accepted_token_count
            else 0.0,
            "sentences": len(split),
            "tokens": sum(len(record["tokens"]) for record in split),
        }
        for name, split in splits.items()
    }
    statistics: dict[str, Any] = {
        "accepted_sentence_count": len(accepted_records),
        "accepted_token_count": accepted_token_count,
        "dataset": DATASET_ID,
        "entity_type_counts": dict(sorted(entity_type_counts.items())),
        "entity_types": sorted(entity_type_counts),
        "label_counts": dict(sorted(label_counts.items())),
        "label_vocabulary": sorted(label_counts),
        "length_summary": _length_summary(accepted_records),
        "rejected_diagnostic_count": len(rejected),
        "rejected_sentence_count": len(rejected_sentence_ids),
        "required_columns": list(REQUIRED_COLUMNS),
        "revision": DATASET_REVISION,
        "seed": seed,
        "source_columns": source_columns,
        "source_row_count": source_row_count,
        "source_sha256": source_sha256,
        "source_sentence_count": len(source_sentence_ids),
        "split_rounding": "floor(0.8*N) train, floor(0.1*N) validation, remainder test",
        "splits": split_statistics,
        "supported_entity_types": list(ENTITY_TYPES),
    }

    _atomic_write_jsonl(output_dir / "train.jsonl", splits["train"])
    _atomic_write_jsonl(output_dir / "validation.jsonl", splits["validation"])
    _atomic_write_jsonl(output_dir / "test.jsonl", splits["test"])
    _atomic_write_json(output_dir / "statistics.json", statistics)
    _atomic_write_jsonl(output_dir / "rejected.jsonl", rejected)
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Local pinned TSV input.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Processed output directory."
    )
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="Sentence split seed.")
    parser.add_argument(
        "--sha256",
        default=DATASET_SHA256,
        help="Expected input SHA-256; pass an empty value only for noncanonical fixtures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = prepare_dataset(
        args.input,
        args.output_dir,
        args.seed,
        expected_sha256=args.sha256 or None,
    )
    print(
        f"Prepared {statistics['accepted_sentence_count']} accepted sentences from "
        f"{statistics['source_row_count']} source rows in {args.output_dir}"
    )


if __name__ == "__main__":
    main()
