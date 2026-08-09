from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import train_ner
from uzbek_speech_entities.ner.labels import validate_bio_record
from uzbek_speech_entities.ner.speech_augmentation import (
    KNOWN_OGG_TRANSCRIPT_FINGERPRINTS,
    authored_templates,
    build_speech_records,
    build_speech_training_file,
    expand_temporal_tokens,
    full_text_fingerprint,
    normalized_ngrams,
    speech_variant,
)
from uzbek_speech_entities.ner.training_config import load_ner_training_config
from uzbek_speech_entities.ner.training_data import select_prepared_data_files


def _record(identifier: str = "source-1") -> dict[str, object]:
    return {
        "id": identifier,
        "tokens": ["Salom", ",", "6-avgust", "kuni", "Akmal", "keldi", "."],
        "ner_tags": ["O", "O", "B-TEMPORAL", "I-TEMPORAL", "B-PER", "O", "O"],
        "source": "fixture",
        "augmentation": None,
    }


def _write(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_spoken_date_and_year_expansion_preserve_strict_bio() -> None:
    tokens, tags, count = expand_temporal_tokens(
        ["16", "avgust", "2026-yil", "18"],
        ["B-TEMPORAL", "I-TEMPORAL", "B-TEMPORAL", "B-NUMERIC"],
    )

    assert tokens == ["oʻn", "oltinchi", "avgust", "ikki", "ming", "yigirma", "olti", "yil", "18"]
    assert tags == [
        "B-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "B-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "I-TEMPORAL",
        "B-NUMERIC",
    ]
    assert count == 2


def test_variant_removes_only_standalone_o_punctuation_and_inserts_safe_filler() -> None:
    tokens, tags, counts = speech_variant(_record())

    assert "," not in tokens and "." not in tokens
    assert "am" in tokens or "um" in tokens or "ee" in tokens
    assert tags[tokens.index("akmal")] == "B-PER"
    assert counts["o_punctuation_removed"] == 2
    validate_bio_record({"id": "variant", "tokens": tokens, "ner_tags": tags})


def test_templates_are_legal_and_include_hard_negatives_without_ogg_corruption() -> None:
    output, statistics = build_speech_records([_record()])

    assert statistics["template_counts"]["hard_negative"] >= 50
    assert len(statistics["allowed_transformations"]) == len(
        set(statistics["allowed_transformations"])
    )
    assert all("doskon" not in " ".join(record["tokens"]) for record in output)
    assert any(
        record["augmentation"].get("template") == "hard_negative"
        for record in output
        if record["augmentation"]
    )
    for record in output:
        validate_bio_record(record)


def test_authored_context_labels_relative_dates_and_semantic_heads_correctly() -> None:
    source = {
        "id": "org-source",
        "tokens": ["Taraqqiyot"],
        "ner_tags": ["B-ORG"],
        "source": "fixture",
        "augmentation": None,
    }
    templates = authored_templates([source])
    bugun = next(record for record in templates if record["id"] == "speech-template-relative-00")
    organization = next(
        record for record in templates if record["id"] == "speech-template-org-boundary-000-00"
    )
    doston_edit = next(
        record for record in templates if record["id"] == "speech-template-per-edit-03"
    )

    assert bugun["tokens"] == ["bugun", "uchrashamiz"]
    assert bugun["ner_tags"] == ["B-TEMPORAL", "O"]
    assert organization["tokens"] == ["taraqqiyot", "tashkiloti", "ishlayman"]
    assert organization["ner_tags"] == ["B-ORG", "I-ORG", "O"]
    assert doston_edit["tokens"] == ["mening", "ismim", "dostoa"]
    assert doston_edit["ner_tags"] == ["O", "O", "B-PER"]


def test_spoken_year_curriculum_is_dense_connective_and_compositional() -> None:
    years = [
        record
        for record in authored_templates([])
        if record["augmentation"]["template"] == "spoken_year"
    ]

    assert len(years) >= 150
    assert any("va" in record["tokens"] for record in years)
    assert any("yilda" in record["tokens"] for record in years)
    assert all(
        record["ner_tags"][record["tokens"].index("yil")].endswith("TEMPORAL")
        if "yil" in record["tokens"]
        else record["ner_tags"][record["tokens"].index("yilda")].endswith("TEMPORAL")
        for record in years
    )


def test_authored_templates_exclude_protected_three_to_five_grams_only() -> None:
    source = {
        "id": "org-source",
        "tokens": ["Taraqqiyot"],
        "ner_tags": ["B-ORG"],
        "source": "fixture",
        "augmentation": None,
    }
    protected = normalized_ngrams("taraqqiyot tashkiloti ishlayman")

    generated, statistics = build_speech_records(
        [source], protected_authored_template_ngrams=protected
    )
    repeated, repeated_statistics = build_speech_records(
        [source], protected_authored_template_ngrams=protected
    )

    authored = [record for record in generated if record["source"] == "speech_ner_template"]
    assert generated == repeated
    assert statistics == repeated_statistics
    assert all(
        not normalized_ngrams(record["tokens"]).intersection(protected) for record in authored
    )
    assert statistics["excluded_protected_authored_template_count"] == 1
    assert (
        statistics["excluded_protected_authored_template_ids_sha256"]
        == hashlib.sha256(b"speech-template-org-boundary-000-00").hexdigest()
    )
    assert statistics["protected_authored_template_ngram_count"] == len(protected)
    assert statistics["protected_authored_template_ngram_overlap_count"] == 0
    assert len(statistics["protected_authored_template_ngrams_sha256"]) == 64


def test_train_derived_boundary_curricula_pair_per_and_loc_contexts_with_legal_bio() -> None:
    source = [
        {
            "id": "org-source",
            "tokens": ["Taraqqiyot"],
            "ner_tags": ["B-ORG"],
            "source": "fixture",
            "augmentation": None,
        },
        {
            "id": "loc-source",
            "tokens": ["Samarqand"],
            "ner_tags": ["B-LOC"],
            "source": "fixture",
            "augmentation": None,
        },
        {
            "id": "per-source",
            "tokens": ["Dilafruz"],
            "ner_tags": ["B-PER"],
            "source": "fixture",
            "augmentation": None,
        },
    ]

    templates = authored_templates(source)
    boundary = [
        record
        for record in templates
        if record["augmentation"]["template"] == "entity_boundary_curriculum"
    ]
    assert {record["tokens"][-2] for record in boundary if record["ner_tags"][0] == "B-ORG"} == {
        "tashkiloti",
        "tashkilotida",
        "kompaniyasi",
        "kompaniyasida",
        "markazi",
        "markazida",
    }
    assert {record["tokens"][-2] for record in boundary if record["ner_tags"][0] == "B-LOC"} == {
        "shahri",
        "shahrida",
        "tumani",
        "tumanida",
        "viloyati",
        "viloyatida",
    }
    assert (
        sum(
            record["augmentation"]["template"] == "entity_head_hard_negative"
            for record in templates
        )
        == 6
    )
    contrast = [
        record
        for record in templates
        if record["augmentation"]["template"] == "eponymic_loc_per_contrast"
    ]
    person = next(record for record in contrast if record["ner_tags"][1] == "B-PER")
    locations = [record for record in contrast if record["ner_tags"][0] == "B-LOC"]
    assert person["tokens"] == ["men", "dilafruz", "bilan", "gaplashdim"]
    assert person["ner_tags"] == ["O", "B-PER", "O", "O"]
    assert len(locations) == 4
    assert all(record["tokens"][0] == person["tokens"][1] for record in locations)
    assert all(record["ner_tags"] == ["B-LOC", "I-LOC", "O"] for record in locations)
    for record in templates:
        validate_bio_record(record)


def test_generation_is_deterministic_and_excludes_protected_source_overlap() -> None:
    first, first_stats = build_speech_records([_record()], seed=77)
    second, second_stats = build_speech_records([_record()], seed=77)

    assert first == second
    assert first_stats == second_stats
    protected_output, protected_stats = build_speech_records(
        [_record()],
        protected_fingerprints=frozenset({full_text_fingerprint(first[0]["tokens"])}),
    )
    assert protected_stats["excluded_protected_source_record_count"] == 1
    assert all(record["id"] != "source-1" for record in protected_output)
    with pytest.raises(ValueError, match="denylisted"):
        build_speech_records(
            [_record()],
            denylisted_fingerprints=frozenset({full_text_fingerprint(first[0]["tokens"])}),
        )


def test_file_builder_keeps_protected_bytes_and_records_provenance(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    fixture = tmp_path / "fixture.jsonl"
    output = tmp_path / "train_speech_augmented.jsonl"
    statistics = tmp_path / "speech_augmentation_statistics.json"
    _write(train, [_record()])
    _write(validation, [{"id": "v", "tokens": ["held", "out"], "ner_tags": ["O", "O"]}])
    _write(fixture, [{"id": "f", "text": "another held out", "entities": []}])
    before = {path: path.read_bytes() for path in (validation, fixture)}

    result = build_speech_training_file(
        train, output, statistics, protected_paths=(validation, fixture)
    )

    assert output.is_file() and statistics.is_file()
    assert result["protected_overlap_count"] == 0
    assert result["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert {path: path.read_bytes() for path in before} == before
    generated = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(
        full_text_fingerprint(record["tokens"]) != full_text_fingerprint("another held out")
        for record in generated
    )


def test_synthetic_records_do_not_match_immutable_speech_fixture() -> None:
    fixture = Path("tests/fixtures/speech_ner_eval.jsonl")
    before = fixture.read_bytes()
    protected = frozenset(
        full_text_fingerprint(json.loads(line)["text"])
        for line in before.decode("utf-8").splitlines()
    )

    generated, _ = build_speech_records([_record()], protected_fingerprints=protected)

    assert fixture.read_bytes() == before
    assert not {full_text_fingerprint(record["tokens"]) for record in generated} & protected
    assert KNOWN_OGG_TRANSCRIPT_FINGERPRINTS
    assert (
        not {full_text_fingerprint(record["tokens"]) for record in generated}
        & KNOWN_OGG_TRANSCRIPT_FINGERPRINTS
    )


def test_custom_augmented_file_and_statistics_names_are_validated_and_selected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "speech.yaml"
    config.write_text(
        Path("configs/ner_speech_candidate.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    loaded = load_ner_training_config(config)

    assert loaded.values["data"]["augmented_train_filename"] == "train_speech_augmented.jsonl"
    assert (
        select_prepared_data_files(tmp_path, True, augmented_train_filename="speech.jsonl")["train"]
        == tmp_path / "speech.jsonl"
    )
    (tmp_path / "statistics.json").write_text(
        json.dumps({"dataset": "fixture", "revision": "v1", "source_sha256": "sha", "seed": 1}),
        encoding="utf-8",
    )
    (tmp_path / "speech_stats.json").write_text(
        json.dumps(
            {
                "allowed_transformations": [],
                "augmented_record_count": 1,
                "output_record_count": 2,
                "seed": 1,
                "selected_source_ids_sha256": "selected",
                "source_record_count": 1,
                "transformation_counts": {},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "speech.jsonl").write_bytes(b'{"id":"speech"}\n')
    monkeypatch.setattr(train_ner, "DEFAULT_DATA_DIR", tmp_path)
    provenance = train_ner._dataset_provenance(True, "speech.jsonl", "speech_stats.json")
    assert (
        provenance["augmentation"]["train_augmented_sha256"]
        == hashlib.sha256(b'{"id":"speech"}\n').hexdigest()
    )
    config.write_text(
        config.read_text(encoding="utf-8").replace("train_speech_augmented.jsonl", "../bad.jsonl"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="basename"):
        load_ner_training_config(config)
