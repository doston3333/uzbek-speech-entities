"""Mine calendar-year-only temporal records from pinned public speech corpora."""

from __future__ import annotations

import argparse
from pathlib import Path

from uzbek_speech_entities.ner.public_corpora import mine_allowlisted_speech


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = mine_allowlisted_speech(args.output, args.stats)
    print(f"Mined {statistics['accepted_records']} public speech year records to {args.output}")


if __name__ == "__main__":
    main()
