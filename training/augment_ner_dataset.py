"""Create deterministic, label-safe augmented NER training data only."""

from __future__ import annotations

import argparse
from pathlib import Path

from uzbek_speech_entities.ner.augmentation import DEFAULT_SEED, augment_training_file

DEFAULT_INPUT = Path("data/processed/ner/train.jsonl")
DEFAULT_OUTPUT = Path("data/processed/ner/train_augmented.jsonl")
DEFAULT_STATISTICS = Path("data/processed/ner/augmentation_statistics.json")


def parse_args() -> argparse.Namespace:
    """Parse paths and deterministic seed without touching validation or test files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistics = augment_training_file(args.input, args.output, args.statistics, args.seed)
    print(
        f"Wrote {statistics['output_record_count']} records "
        f"({statistics['augmented_record_count']} augmented) to {args.output}"
    )


if __name__ == "__main__":
    main()
