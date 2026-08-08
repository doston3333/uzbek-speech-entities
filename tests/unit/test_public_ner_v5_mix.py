from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from uzbek_speech_entities.ner.public_mix import build_public_v5_mix
from uzbek_speech_entities.ner.speech_augmentation import full_text_fingerprint


def _record(identifier: str, tokens: list[str], source: str = "fixture") -> dict[str, object]:
    return {
        "id": identifier,
        "tokens": tokens,
        "ner_tags": ["B-TEMPORAL", *(["I-TEMPORAL"] * (len(tokens) - 1))],
        "source": source,
        "augmentation": None,
    }


def _write(path: Path, records: list[dict[str, object]]) -> bytes:
    value = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(value, encoding="utf-8")
    return value.encode("utf-8")


def _read(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_v5_mix_is_deterministic_capped_and_keeps_speech_dev_out_of_train(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.jsonl"
    expert = tmp_path / "expert.jsonl"
    temporal = tmp_path / "temporal.jsonl"
    speech_one = tmp_path / "speech-one.jsonl"
    speech_two = tmp_path / "speech-two.jsonl"
    protected = tmp_path / "protected.jsonl"
    base_bytes = _write(base, [_record("base", ["ikki", "ming", "yil"])])
    _write(
        expert,
        [
            _record("expert-duplicate", ["ikki", "ming", "yil"]),
            _record("expert-a", ["bir", "ming", "yil"]),
            _record("expert-b", ["uch", "ming", "yil"]),
        ],
    )
    _write(
        temporal,
        [
            _record("temporal-protected", ["toʻrt", "ming", "yil"]),
            _record("temporal-a", ["besh", "ming", "yil"]),
            _record("temporal-b", ["olti", "ming", "yil"]),
        ],
    )
    _write(
        speech_one,
        [
            _record("speech-a", ["bir", "ming", "toʻqqiz", "yuz", "yili"], "cv"),
            _record("speech-b", ["ikki", "ming", "va", "besh", "yilni"], "cv"),
        ],
    )
    _write(
        speech_two,
        [
            _record("speech-c", ["ikki", "ming", "yigirma", "yilda"], "usc"),
            _record("speech-d", ["ikki", "ming", "oʻn", "yildan"], "usc"),
        ],
    )
    protected_bytes = _write(
        protected, [_record("held-out", ["toʻrt", "ming", "yil"])]
    )

    def build(prefix: str) -> tuple[list[dict[str, object]], list[dict[str, object]], dict]:
        output = tmp_path / f"{prefix}-train.jsonl"
        stats_path = tmp_path / f"{prefix}-stats.json"
        dev = tmp_path / f"{prefix}-dev.jsonl"
        stats = build_public_v5_mix(
            base,
            expert,
            temporal,
            (speech_one, speech_two),
            output,
            stats_path,
            dev,
            protected_paths=(protected,),
            seed=77,
            expert_cap=1,
            temporal_cap=1,
            speech_train_cap=2,
            core_upsample_factors={},
            base_loc_upsample_factor=1,
            hard_loc_upsample_factor=1,
            hard_org_upsample_factor=1,
            speech_dev_fraction=0.5,
            minimum_speech_dev_records=2,
            minimum_speech_dev_sources=2,
        )
        return _read(output), _read(dev), stats

    first_train, first_dev, first_stats = build("first")
    second_train, second_dev, second_stats = build("second")
    assert first_train == second_train
    assert first_dev == second_dev
    assert first_stats["selected_source_ids_sha256"] == second_stats[
        "selected_source_ids_sha256"
    ]
    assert len(first_train) == 5  # base + one expert + one temporal + two speech-train.
    assert len(first_dev) == 2  # one held out from each public speech source.
    train_fingerprints = {full_text_fingerprint(record["tokens"]) for record in first_train}
    dev_fingerprints = {full_text_fingerprint(record["tokens"]) for record in first_dev}
    assert train_fingerprints.isdisjoint(dev_fingerprints)
    assert full_text_fingerprint(["toʻrt", "ming", "yil"]) not in train_fingerprints
    assert first_stats["excluded_counts"] == {
        "uzner_expert:normalized_duplicate": 1,
        "uzner_temporal:protected_exact_text": 1,
    }
    assert any(
        "digit calendar years" in transformation
        for transformation in first_stats["allowed_transformations"]
    )
    assert base.read_bytes() == base_bytes
    assert protected.read_bytes() == protected_bytes


def test_v5_mix_refuses_base_overlap_with_denylist(tmp_path: Path) -> None:
    paths = [tmp_path / name for name in ("base", "expert", "temporal", "speech")]
    _write(paths[0], [_record("base", ["secret", "year"])])
    for path in paths[1:]:
        _write(path, [])
    protected = tmp_path / "protected"
    _write(protected, [])
    fingerprint = full_text_fingerprint(["secret", "year"])
    with pytest.raises(ValueError, match="known OGG denylist"):
        build_public_v5_mix(
            paths[0],
            paths[1],
            paths[2],
            (paths[3],),
            tmp_path / "out",
            tmp_path / "stats",
            tmp_path / "dev",
            protected_paths=(protected,),
            denylisted_fingerprints=frozenset({fingerprint}),
        )


def test_v5_mix_statistics_sha_matches_written_output(tmp_path: Path) -> None:
    base, expert, temporal, speech, protected = [
        tmp_path / name for name in ("base", "expert", "temporal", "speech", "protected")
    ]
    _write(base, [_record("base", ["ikki", "ming", "yil"])])
    for path in (expert, temporal, protected):
        _write(path, [])
    _write(
        speech,
        [
            _record("speech-a", ["bir", "ming", "yili"], "cv"),
            _record("speech-b", ["ikki", "ming", "yili"], "cv"),
            _record("speech-c", ["uch", "ming", "yili"], "usc"),
            _record("speech-d", ["toʻrt", "ming", "yili"], "usc"),
        ],
    )
    output, stats_path, dev = tmp_path / "out", tmp_path / "stats", tmp_path / "dev"
    stats = build_public_v5_mix(
        base,
        expert,
        temporal,
        (speech,),
        output,
        stats_path,
        dev,
        protected_paths=(protected,),
        speech_dev_fraction=0.5,
        minimum_speech_dev_records=2,
        minimum_speech_dev_sources=2,
    )
    assert stats["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert stats["speech_dev_sha256"] == hashlib.sha256(dev.read_bytes()).hexdigest()


def test_v5_mix_refuses_an_empty_speech_dev_gate(tmp_path: Path) -> None:
    base, expert, temporal, speech, protected = [
        tmp_path / name for name in ("base", "expert", "temporal", "speech", "protected")
    ]
    _write(base, [_record("base", ["ikki", "ming", "yil"])])
    for path in (expert, temporal, speech, protected):
        _write(path, [])
    with pytest.raises(ValueError, match="dev set is too small"):
        build_public_v5_mix(
            base,
            expert,
            temporal,
            (speech,),
            tmp_path / "out",
            tmp_path / "stats",
            tmp_path / "dev",
            protected_paths=(protected,),
        )


def test_v5_mix_requires_speech_dev_source_coverage(tmp_path: Path) -> None:
    base, expert, temporal, speech, protected = [
        tmp_path / name for name in ("base", "expert", "temporal", "speech", "protected")
    ]
    _write(base, [_record("base", ["ikki", "ming", "yil"])])
    for path in (expert, temporal, protected):
        _write(path, [])
    _write(
        speech,
        [
            _record(f"speech-{index}", [str(index), "yili"], "only-one-source")
            for index in range(20)
        ],
    )
    with pytest.raises(ValueError, match="lacks source coverage"):
        build_public_v5_mix(
            base,
            expert,
            temporal,
            (speech,),
            tmp_path / "out",
            tmp_path / "stats",
            tmp_path / "dev",
            protected_paths=(protected,),
            minimum_speech_dev_records=2,
            minimum_speech_dev_sources=2,
        )


def test_v5_statistics_are_byte_stable_across_temporary_roots(tmp_path: Path) -> None:
    statistics: list[bytes] = []
    for root_name in ("one", "two"):
        root = tmp_path / root_name
        root.mkdir()
        base, expert, temporal, speech_one, speech_two, protected = [
            root / name
            for name in ("base", "expert", "temporal", "speech-one", "speech-two", "protected")
        ]
        _write(base, [_record("base", ["ikki", "ming", "yil"])])
        for path in (expert, temporal, protected):
            _write(path, [])
        _write(
            speech_one,
            [
                _record("cv-a", ["bir", "ming", "yili"], "cv"),
                _record("cv-b", ["ikki", "ming", "yili"], "cv"),
            ],
        )
        _write(
            speech_two,
            [
                _record("usc-a", ["uch", "ming", "yili"], "usc"),
                _record("usc-b", ["toʻrt", "ming", "yili"], "usc"),
            ],
        )
        build_public_v5_mix(
            base,
            expert,
            temporal,
            (speech_one, speech_two),
            root / "train.jsonl",
            root / "statistics.json",
            root / "dev.jsonl",
            protected_paths=(protected,),
            speech_dev_fraction=0.5,
            minimum_speech_dev_records=2,
            minimum_speech_dev_sources=2,
        )
        statistics.append((root / "statistics.json").read_bytes())
    assert statistics[0] == statistics[1]
