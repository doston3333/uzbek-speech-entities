from __future__ import annotations

from pathlib import Path

import pytest

from training import modal_public_ner_v5


def test_v5_modal_seed_selection_is_parallel_safe_and_whitelisted() -> None:
    assert modal_public_ner_v5.selected_config_basenames("seed1") == (
        "ner_public_v5_ft24_lr4e7_ep2_p1_seed1.yaml",
    )
    assert modal_public_ner_v5.selected_config_basenames("seed2") == (
        "ner_public_v5_ft24_lr4e7_ep2_p1_seed2.yaml",
    )
    assert modal_public_ner_v5.selected_config_basenames(
        "both"
    ) == modal_public_ner_v5.selected_config_basenames("sweep")
    assert len(modal_public_ner_v5.selected_config_basenames("sweep")) == 16
    outputs = {
        value.output_name for value in modal_public_ner_v5.APPROVED_CONFIGS.values()
    }
    assert len(outputs) == 16
    with pytest.raises(ValueError, match="seed must be"):
        modal_public_ner_v5.selected_config_basenames("seed3")


def test_v5_remote_config_rewrites_training_and_output() -> None:
    original = {
        "model": {"checkpoint": "./models/ner/final", "max_length": 128},
        "training": {
            "seed": 1,
            "learning_rate": 1.0,
            "epochs": 99,
            "weight_decay": 0.99,
        },
        "data": {"augmented_train_filename": "train_public_v5.jsonl"},
        "output": {"directory": "local"},
    }
    rewritten = modal_public_ner_v5.rewritten_remote_config(
        "ner_public_v5_ft24_lr4e7_ep2_p1_seed1.yaml", original
    )
    assert rewritten["model"]["checkpoint"] == "/workspace/models/ner/final"
    assert rewritten["output"]["directory"] == (
        "/outputs/public-ner-v5n24-20260808/public-v5n24-ft24-lr4e7-ep2-p1-seed1"
    )
    assert rewritten["training"] == {
        "seed": 20261063,
        "learning_rate": 0.0000004,
        "epochs": 2,
        "weight_decay": 0.01,
    }
    with pytest.raises(ValueError, match="approved public V5"):
        modal_public_ner_v5.rewritten_remote_config("unapproved.yaml", original)


def test_v5_public_sources_are_exactly_pinned() -> None:
    zero = modal_public_ner_v5.speech_corpus("zero-shot-news")
    assert zero.revision == "a4ad4607a7c1ffe492fe1420a05fb6c7b6165383"
    with pytest.raises(ValueError, match="speech source"):
        modal_public_ner_v5.speech_corpus("other")


def test_v5_release_name_cannot_escape_volume_prefix() -> None:
    assert modal_public_ner_v5.validate_release_name("public-ner-v5-20260807") == (
        "public-ner-v5-20260807"
    )
    with pytest.raises(ValueError, match="release"):
        modal_public_ner_v5.validate_release_name("../escape")


def test_parallel_result_collection_waits_for_every_call_and_aggregates_failures() -> None:
    class Call:
        def __init__(self, result: object = None, error: Exception | None = None):
            self.result = result
            self.error = error

        def get(self) -> object:
            if self.error:
                raise self.error
            return self.result

    assert modal_public_ner_v5.collect_call_results(
        [("one", Call(1)), ("two", Call(2))]
    ) == [1, 2]
    with pytest.raises(RuntimeError, match=r"one: ValueError"):
        modal_public_ner_v5.collect_call_results(
            [("one", Call(error=ValueError("bad")))]
        )


def test_publication_manifest_is_last_complete_and_tamper_evident(tmp_path: Path) -> None:
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    (local / "records.jsonl").write_text("{}\n", encoding="utf-8")
    (local / "statistics.json").write_text("{}\n", encoding="utf-8")
    manifest = modal_public_ner_v5._publish_directory(local, remote, {"job": "fixture"})
    assert manifest["complete"] is True
    (remote / "records.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        modal_public_ner_v5._verify_manifest(remote)


def test_v5_bundle_has_labels_and_parent_hash_is_exact() -> None:
    assert "labels.json" in modal_public_ner_v5.INFERENCE_BUNDLE_FILES
    assert modal_public_ner_v5.EXPECTED_PARENT_MODEL_SHA256 == (
        "498d6bf27179d49af48cfb264bd6eb5cf534f9ed507eaf4435e275157888b976"
    )
    assert modal_public_ner_v5.LOCAL_PARENT_DIR.name == (
        "public-v5n5-ft5-lr3e6-ep1-seed2"
    )
    assert modal_public_ner_v5.DEFAULT_RELEASE == "public-ner-v5n24-20260808"


def test_cached_seed_must_match_requested_data_release() -> None:
    release_manifest = {"complete": True, "files": {"train": {"sha256": "a"}}}
    digest = modal_public_ner_v5._manifest_digest(release_manifest)
    modal_public_ner_v5._verify_release_binding(
        {
            "data_release": "release-a",
            "data_release_manifest_sha256": digest,
        },
        "release-a",
        release_manifest,
    )
    with pytest.raises(ValueError, match="different data release"):
        modal_public_ner_v5._verify_release_binding(
            {
                "data_release": "release-b",
                "data_release_manifest_sha256": digest,
            },
            "release-a",
            release_manifest,
        )
