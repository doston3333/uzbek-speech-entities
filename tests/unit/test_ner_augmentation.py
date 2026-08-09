from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import train_ner
from uzbek_speech_entities.ner.augmentation import (
    TRANSFORMATION_ORDER,
    apply_transformations,
    augment_records,
    augment_training_file,
)
from uzbek_speech_entities.ner.labels import validate_bio_record


def _record(index: int) -> dict[str, object]:
    return {
        "id": f"sentence-{index:06d}",
        "tokens": ["Akmal", "soat", "01:29da", "."],
        "ner_tags": ["B-PER", "O", "B-TEMPORAL", "O"],
        "source": "fixture",
        "augmentation": None,
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    path.write_text(
        content,
        encoding="utf-8",
    )


def test_transformation_requests_reject_invalid_combinations_and_report_noops() -> None:
    with pytest.raises(ValueError, match="unknown"):
        apply_transformations(["Akmal"], ["B-PER"], ["spelling_noise"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_transformations(["Akmal"], ["B-PER"], ["lowercase", "reduce_capitalization"])

    tokens, tags, applied = apply_transformations(
        ["already", "lowercase"], ["O", "O"], ["lowercase"]
    )

    assert tokens == ["already", "lowercase"]
    assert tags == ["O", "O"]
    assert applied == ()


def test_apostrophes_flip_only_inside_words() -> None:
    tokens, tags, applied = apply_transformations(
        ["'", "o'z", "o’z", "oʻz", "oʼz", "o‘z", "o`z"],
        ["O"] * 7,
        ["apostrophe_representation"],
    )

    assert tokens == ["'", "oʻz", "o'z", "o'z", "o'z", "o'z", "o'z"]
    assert tags == ["O"] * 7
    assert applied == ("apostrophe_representation",)


def test_punctuation_removal_keeps_internal_and_entity_punctuation() -> None:
    tokens, tags, applied = apply_transformations(
        [".", "salom", ",", "Akmal", "!"],
        ["O", "O", "O", "B-PER", "O"],
        ["remove_punctuation"],
    )

    assert tokens == ["salom", ",", "Akmal"]
    assert tags == ["O", "O", "B-PER"]
    assert applied == ("remove_punctuation",)

    tokens, tags, applied = apply_transformations(
        [",", "Akmal", "."],
        ["B-MISC", "I-MISC", "O"],
        ["remove_punctuation"],
    )

    assert tokens == [",", "Akmal"]
    assert tags == ["B-MISC", "I-MISC"]
    assert applied == ("remove_punctuation",)


def test_hyphen_and_colon_spacing_preserve_legal_bio_sequences() -> None:
    tokens, tags, applied = apply_transformations(
        ["ilm-fan", "2026-yil", "01:29:44", "01:29da"],
        ["O", "B-TEMPORAL", "B-TEMPORAL", "B-TEMPORAL"],
        ["hyphen_spacing", "colon_spacing"],
    )

    assert tokens == ["ilm-fan", "2026", "yil", "01", ":", "29", ":", "44", "01", ":", "29da"]
    assert tags == [
        "O",
        "B-TEMPORAL",
        "I-TEMPORAL",
        "B-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "B-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
    ]
    assert applied == ("hyphen_spacing", "colon_spacing")

    tokens, tags, applied = apply_transformations(
        ["Akmal", "-"], ["B-PER", "I-PER"], ["hyphen_spacing"]
    )

    assert tokens == ["Akmal", "-"]
    assert tags == ["B-PER", "I-PER"]
    assert applied == ()


def test_augmentation_ratio_metadata_and_statistics_are_deterministic() -> None:
    records = [_record(index) for index in range(7)]

    first_records, first_statistics = augment_records(records, seed=42)
    second_records, second_statistics = augment_records(records, seed=42)

    assert first_records == second_records
    assert first_statistics == second_statistics
    assert first_records[:7] == records
    assert len(first_records) == 10
    assert first_statistics["source_record_count"] == 7
    assert first_statistics["augmented_record_count"] == 3
    assert first_statistics["output_record_count"] == 10
    assert first_statistics["augmentation_fraction_of_output"] == 0.3
    assert first_statistics["allowed_transformations"] == list(TRANSFORMATION_ORDER)
    assert len(first_statistics["selected_source_ids_sha256"]) == 64
    augmented = first_records[7:]
    source_by_id = {record["id"]: record for record in records}
    source_ids = [record["augmentation"]["source_id"] for record in augmented]
    assert len(set(source_ids)) == 3
    for record in augmented:
        metadata = record["augmentation"]
        assert record["id"].startswith(f"{metadata['source_id']}__aug_")
        assert set(metadata["transformations"]) <= set(TRANSFORMATION_ORDER)
        assert metadata["transformations"] == sorted(
            metadata["transformations"], key=TRANSFORMATION_ORDER.index
        )
        assert metadata["transformations"]
        source = source_by_id[metadata["source_id"]]
        assert (record["tokens"], record["ner_tags"]) != (
            source["tokens"],
            source["ner_tags"],
        )
        validate_bio_record(record)


def test_augmentation_rejects_existing_augmentation_metadata() -> None:
    record = _record(1)
    record["augmentation"] = {"source_id": "sentence-000000", "transformations": ["lowercase"]}

    with pytest.raises(ValueError, match="already augmented"):
        augment_records([record])


def test_file_augmentation_leaves_validation_and_test_bytes_unchanged(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "train_augmented.jsonl"
    statistics_path = tmp_path / "augmentation_statistics.json"
    validation_path = tmp_path / "validation.jsonl"
    test_path = tmp_path / "test.jsonl"
    _write_jsonl(train_path, [_record(index) for index in range(7)])
    validation_bytes = b'{"validation":"sentinel"}\n'
    test_bytes = b'{"test":"sentinel"}\n'
    validation_path.write_bytes(validation_bytes)
    test_path.write_bytes(test_bytes)

    statistics = augment_training_file(train_path, output_path, statistics_path, seed=42)

    assert statistics["output_record_count"] == 10
    assert output_path.is_file()
    assert statistics_path.is_file()
    assert validation_path.read_bytes() == validation_bytes
    assert test_path.read_bytes() == test_bytes


def test_file_augmentation_rejects_protected_or_missing_paths(tmp_path: Path) -> None:
    input_path = tmp_path / "train.jsonl"
    output_path = tmp_path / "train_augmented.jsonl"

    with pytest.raises(ValueError, match="input and output"):
        augment_training_file(input_path, input_path)
    with pytest.raises(ValueError, match="statistics path must differ"):
        augment_training_file(input_path, output_path, input_path)
    with pytest.raises(ValueError, match="statistics path must differ"):
        augment_training_file(input_path, output_path, output_path)
    with pytest.raises(ValueError, match="statistics path"):
        augment_training_file(input_path, output_path, tmp_path / "validation.jsonl")
    with pytest.raises(FileNotFoundError, match="missing source training JSONL"):
        augment_training_file(input_path, output_path)


def test_augmented_dataset_provenance_archives_statistics_and_corpus_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_statistics = {
        "dataset": "fixture/dataset",
        "revision": "abc123",
        "source_sha256": "source-sha",
        "seed": 42,
    }
    augmentation_statistics = {
        "allowed_transformations": list(TRANSFORMATION_ORDER),
        "augmented_record_count": 3,
        "output_record_count": 10,
        "seed": 42,
        "selected_source_ids_sha256": "selected-sha",
        "source_record_count": 7,
        "transformation_counts": {name: 1 for name in TRANSFORMATION_ORDER},
    }
    augmented_bytes = b'{"id":"fixture"}\n'
    (tmp_path / "statistics.json").write_text(json.dumps(base_statistics), encoding="utf-8")
    (tmp_path / "augmentation_statistics.json").write_text(
        json.dumps(augmentation_statistics), encoding="utf-8"
    )
    (tmp_path / "train_augmented.jsonl").write_bytes(augmented_bytes)
    monkeypatch.setattr(train_ner, "DEFAULT_DATA_DIR", tmp_path)

    provenance = train_ner._dataset_provenance(augmentation=True)

    assert provenance["dataset"] == "fixture/dataset"
    assert provenance["augmentation"]["augmented_record_count"] == 3
    assert (
        provenance["augmentation"]["train_augmented_sha256"]
        == hashlib.sha256(augmented_bytes).hexdigest()
    )
