from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.prepare_ner_dataset import REQUIRED_COLUMNS, prepare_dataset


def _write_tsv(path: Path, rows: list[list[str]]) -> None:
    path.write_text(
        "\t".join(REQUIRED_COLUMNS) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_preparation_reconstructs_sorted_sentences_and_writes_required_artifacts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tsv"
    _write_tsv(
        source,
        [
            ["1", "2", "Karimov", "I-PER", "NOUN"],
            ["1", "1", "Akmal", "B-PER", "PROPN"],
            ["2", "1", "Toshkent", "B-LOC", "PROPN"],
            ["3", "1", "kecha", "B-TEMPORAL", "ADV"],
            ["4", "1", "so'm", "B-MONEY", "NOUN"],
            ["5", "1", "42", "B-NUMERIC", "NUM"],
            ["6", "1", "kitob", "B-WORK", "NOUN"],
            ["7", "1", "uyushma", "B-ORG", "NOUN"],
            ["8", "1", "boshqa", "B-MISC", "NOUN"],
            ["9", "1", "gap", "O", "NOUN"],
            ["10", "1", "yana", "O", "NOUN"],
        ],
    )
    output = tmp_path / "processed"

    statistics = prepare_dataset(source, output)

    assert statistics["source_row_count"] == 11
    assert statistics["source_columns"] == list(REQUIRED_COLUMNS)
    assert statistics["accepted_sentence_count"] == 10
    assert statistics["rejected_sentence_count"] == 0
    assert {name: values["sentences"] for name, values in statistics["splits"].items()} == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
    assert sum(values["tokens"] for values in statistics["splits"].values()) == 11
    assert all(
        values["actual_sentence_ratio"] in {0.8, 0.1}
        for values in statistics["splits"].values()
    )
    assert {path.name for path in output.iterdir()} == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "statistics.json",
        "rejected.jsonl",
    }
    records = [
        record
        for name in ("train", "validation", "test")
        for record in _read_jsonl(output / f"{name}.jsonl")
    ]
    sentence_one = next(record for record in records if record["id"] == "sentence-000001")
    assert sentence_one["tokens"] == ["Akmal", "Karimov"]
    assert sentence_one["ner_tags"] == ["B-PER", "I-PER"]
    assert sentence_one["source"] == "uznlp-uz/uzbek_NER"
    assert sentence_one["augmentation"] is None


def test_preparation_rejects_corrupted_sentences_without_partial_acceptance(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    _write_tsv(
        source,
        [
            ["1", "1", "good", "O", "NOUN"],
            ["1", "1", "duplicate", "O", "NOUN"],
            ["2", "1", "broken", "I-PER", "NOUN"],
            ["3", "0", "bad-order", "O", "NOUN"],
            ["3", "1", "must-not-survive", "O", "NOUN"],
            ["4", "1", "valid", "B-LOC", "NOUN"],
            ["", "1", "missing-id", "O", "NOUN"],
        ],
    )
    output = tmp_path / "processed"

    statistics = prepare_dataset(source, output)

    records = [
        record
        for name in ("train", "validation", "test")
        for record in _read_jsonl(output / f"{name}.jsonl")
    ]
    assert [record["id"] for record in records] == ["sentence-000004"]
    assert statistics["rejected_sentence_count"] == 3
    rejected = _read_jsonl(output / "rejected.jsonl")
    assert any(item["reason"] == "duplicate TokenOrder values: [1]" for item in rejected)
    assert any("incompatible or orphan I-PER" in item["reason"] for item in rejected)
    assert any(item["reason"] == "nonpositive TokenOrder: '0'" for item in rejected)
    assert any(item["reason"] == "missing Sentence" for item in rejected)
    assert any(item.get("row", {}).get("Token") == "bad-order" for item in rejected)
    assert any(item.get("record", {}).get("tokens") == ["broken"] for item in rejected)


def test_preparation_is_deterministic_and_splits_are_disjoint(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    _write_tsv(
        source,
        [[str(sentence), "1", f"token-{sentence}", "O", "NOUN"] for sentence in range(1, 22)],
    )
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"

    first_statistics = prepare_dataset(source, first_output)
    second_statistics = prepare_dataset(source, second_output)

    assert first_statistics == second_statistics
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "statistics.json",
        "rejected.jsonl",
    ):
        assert (first_output / filename).read_bytes() == (second_output / filename).read_bytes()
    split_ids = {
        name: {record["id"] for record in _read_jsonl(first_output / f"{name}.jsonl")}
        for name in ("train", "validation", "test")
    }
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert sum(len(ids) for ids in split_ids.values()) == 21


def test_preparation_rejects_an_unexpected_source_schema(tmp_path: Path) -> None:
    source = tmp_path / "wrong-schema.tsv"
    source.write_text("Sentence\tToken\tNER_Tag\tpos\n1\tAkmal\tB-PER\tPROPN\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected TSV columns"):
        prepare_dataset(source, tmp_path / "processed")


def test_preparation_verifies_and_records_source_checksum(tmp_path: Path) -> None:
    source = tmp_path / "source.tsv"
    _write_tsv(source, [["1", "1", "Akmal", "B-PER", "PROPN"]])

    with pytest.raises(ValueError, match="source checksum mismatch"):
        prepare_dataset(source, tmp_path / "bad", expected_sha256="0" * 64)

    statistics = prepare_dataset(source, tmp_path / "good")

    assert len(statistics["source_sha256"]) == 64
