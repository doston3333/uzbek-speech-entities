from __future__ import annotations

import pytest

from training import modal_train_ner


def test_modal_seed_selection_is_whitelisted_and_deterministic() -> None:
    assert modal_train_ner.selected_config_basenames("seed1") == (
        "ner_speech_continuation_v4_seed1.yaml",
    )
    assert modal_train_ner.selected_config_basenames("seed2") == (
        "ner_speech_continuation_v4_seed2.yaml",
    )
    assert modal_train_ner.selected_config_basenames("both") == (
        "ner_speech_continuation_v4_seed1.yaml",
        "ner_speech_continuation_v4_seed2.yaml",
    )
    with pytest.raises(ValueError, match="seed must be"):
        modal_train_ner.selected_config_basenames("unexpected")


def test_remote_config_rewrites_only_checkpoint_and_output_paths() -> None:
    original = {
        "model": {"checkpoint": "./models/ner/final", "max_length": 128},
        "training": {"seed": 20260811, "save_only_model": True},
        "data": {"augmented_train_filename": "train_speech_augmented_v3.jsonl"},
        "output": {"directory": "./models/ner/speech-continuation-20260807-v4-seed1"},
    }

    rewritten = modal_train_ner.rewritten_remote_config(
        "ner_speech_continuation_v4_seed1.yaml", original
    )

    assert original["model"]["checkpoint"] == "./models/ner/final"
    assert original["output"]["directory"] == (
        "./models/ner/speech-continuation-20260807-v4-seed1"
    )
    assert rewritten["model"]["checkpoint"] == str(modal_train_ner.REMOTE_CHECKPOINT)
    assert rewritten["output"]["directory"] == (
        "/outputs/speech-continuation-20260807-v4-seed1"
    )
    assert rewritten["training"] == original["training"]
    assert rewritten["data"] == original["data"]
    with pytest.raises(ValueError, match="approved continuation config"):
        modal_train_ner.rewritten_remote_config("other.yaml", original)


def test_modal_checkpoint_upload_set_contains_inference_files_only() -> None:
    assert modal_train_ner.INFERENCE_CHECKPOINT_FILES == {
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
    assert "optimizer.pt" not in modal_train_ner.INFERENCE_CHECKPOINT_FILES
    assert "trainer_state.json" not in modal_train_ner.INFERENCE_CHECKPOINT_FILES


def test_modal_parent_is_the_promoted_final_checkpoint() -> None:
    assert modal_train_ner.REMOTE_CHECKPOINT.as_posix() == "/workspace/models/ner/final"
    assert modal_train_ner.EXPECTED_PARENT_MODEL_SHA256 == (
        "af2993de66f7a36b4ff5c8b6bd68e08f04183e56e9ea160a745238f3d06ed2a0"
    )
