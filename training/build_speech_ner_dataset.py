"""Build deterministic train-only speech-aware NER augmentation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from uzbek_speech_entities.ner.speech_augmentation import (
    DEFAULT_PROTECTED_PATHS,
    DEFAULT_SEED,
    build_speech_training_file,
)

DEFAULT_INPUT = Path("data/processed/ner/train.jsonl")
DEFAULT_OUTPUT = Path("data/processed/ner/train_speech_augmented.jsonl")
DEFAULT_STATISTICS = Path("data/processed/ner/speech_augmentation_statistics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATISTICS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--protected", type=Path, action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protected = tuple(args.protected) if args.protected else DEFAULT_PROTECTED_PATHS
    statistics = build_speech_training_file(
        args.input, args.output, args.statistics, seed=args.seed, protected_paths=protected
    )
    print(f"Wrote {statistics['output_record_count']} records to {args.output}")


if __name__ == "__main__":
    main()
