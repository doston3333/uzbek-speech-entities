"""Build a deterministic precision-first name lexicon from prepared BIO JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from uzbek_speech_entities.ner.offset_tokens import comparison_key


def build_name_lexicon(
    records: Iterable[dict[str, object]], denylist: frozenset[str] = frozenset()
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Keep normalized tokens with at least one PER tag and >= 60% PER usage."""
    total: Counter[str] = Counter()
    per: Counter[str] = Counter()
    for record in records:
        tokens = record.get("tokens")
        tags = record.get("ner_tags")
        if not isinstance(tokens, list) or not isinstance(tags, list) or len(tokens) != len(tags):
            raise ValueError("prepared records require equally sized tokens and ner_tags lists")
        for token, tag in zip(tokens, tags, strict=True):
            if not isinstance(token, str) or not isinstance(tag, str):
                raise ValueError("prepared tokens and ner_tags must be strings")
            key = comparison_key(token)
            if not key:
                continue
            total[key] += 1
            if tag in {"B-PER", "I-PER"}:
                per[key] += 1
    names = {
        key: {
            "per_count": count,
            "total_count": total[key],
            "per_ratio": round(count / total[key], 6),
        }
        for key, count in sorted(per.items())
        if count >= 1 and count / total[key] >= 0.60 and key not in denylist
    }
    return {"names": names}


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} must be an object")
        records.append(value)
    return records


def _denylist(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    return frozenset(
        comparison_key(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/ner/train.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/uzbek_speech_entities/ner/resources/name_lexicon.json"),
    )
    parser.add_argument("--denylist", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_name_lexicon(_read_jsonl(args.input), _denylist(args.denylist))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
