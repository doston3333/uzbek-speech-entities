"""Import the pinned public UzNER workbook into canonical prepared JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from uzbek_speech_entities.ner.public_corpora import import_uzner_workbook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--mode", choices=("expert", "temporal"), required=True)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = import_uzner_workbook(
        args.workbook,
        args.output,
        args.stats,
        mode=args.mode,
        exclude_jsonl_paths=args.exclude_jsonl,
    )
    print(f"Imported {statistics['accepted_records']} UzNER records to {args.output}")


if __name__ == "__main__":
    main()
