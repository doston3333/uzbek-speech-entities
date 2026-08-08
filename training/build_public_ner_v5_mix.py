"""Build the deterministic public-data V5 NER continuation mix."""

from __future__ import annotations

import argparse
from pathlib import Path

from uzbek_speech_entities.ner.public_mix import build_public_v5_mix
from uzbek_speech_entities.ner.speech_augmentation import DEFAULT_PROTECTED_PATHS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--temporal", type=Path, required=True)
    parser.add_argument("--speech", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--speech-dev", type=Path, required=True)
    parser.add_argument("--protected", type=Path, action="append")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--expert-cap", type=int, default=3000)
    parser.add_argument("--temporal-cap", type=int, default=1500)
    parser.add_argument("--minimum-speech-dev-records", type=int, default=10)
    parser.add_argument("--minimum-speech-dev-sources", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protected = tuple(args.protected) if args.protected else DEFAULT_PROTECTED_PATHS
    statistics = build_public_v5_mix(
        args.base,
        args.expert,
        args.temporal,
        args.speech,
        args.output,
        args.stats,
        args.speech_dev,
        protected_paths=protected,
        seed=args.seed,
        expert_cap=args.expert_cap,
        temporal_cap=args.temporal_cap,
        minimum_speech_dev_records=args.minimum_speech_dev_records,
        minimum_speech_dev_sources=args.minimum_speech_dev_sources,
    )
    print(
        f"Wrote {statistics['output_record_count']} training records and "
        f"{statistics['speech_dev_record_count']} held-out speech records"
    )


if __name__ == "__main__":
    main()
