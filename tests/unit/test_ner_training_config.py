from __future__ import annotations

from pathlib import Path

import pytest

from uzbek_speech_entities.ner.labels import CANONICAL_BIO_LABELS, build_label_maps
from uzbek_speech_entities.ner.training_config import load_ner_training_config
from uzbek_speech_entities.ner.training_data import (
    load_prepared_datasets,
    select_prepared_data_files,
)


def test_label_maps_are_stable_and_preserve_all_supported_bio_labels() -> None:
    label2id, id2label = build_label_maps(["O", "B-PER", "I-WORK"])

    assert tuple(label2id) == CANONICAL_BIO_LABELS
    assert label2id["O"] == 0
    assert len(label2id) == len(id2label) == 17
    assert set(id2label.values()) == set(CANONICAL_BIO_LABELS)


def test_training_config_rejects_wrong_best_model_metric(tmp_path: Path) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
model: {checkpoint: local, max_length: 128}
training:
  seed: 42
  learning_rate: 0.00002
  epochs: 5
  train_batch_size: 8
  eval_batch_size: 16
  gradient_accumulation_steps: 2
  weight_decay: 0.01
  warmup_ratio: 0.05
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: token_accuracy
  greater_is_better: true
  fp16: false
  bf16: false
  dataloader_num_workers: 0
data: {augmentation: false}
output: {directory: ./models/ner/clean}
""".lstrip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metric_for_best_model"):
        load_ner_training_config(config)


def test_augmented_train_file_selection_changes_only_the_train_split(tmp_path: Path) -> None:
    selected = select_prepared_data_files(tmp_path, augmentation=True)

    assert selected == {
        "train": tmp_path / "train_augmented.jsonl",
        "validation": tmp_path / "validation.jsonl",
        "test": tmp_path / "test.jsonl",
    }


def test_augmented_loader_fails_clearly_when_the_train_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="augmented training JSONL is required"):
        load_prepared_datasets(tmp_path, augmentation=True)


@pytest.mark.parametrize(
    ("filename", "seed", "directory"),
    (
        ("ner_speech_candidate_v2_seed1.yaml", 20260807, "speech-candidate-20260807-v2-seed1"),
        ("ner_speech_candidate_v2_seed2.yaml", 20260808, "speech-candidate-20260807-v2-seed2"),
    ),
)
def test_v2_speech_configs_select_isolated_outputs_and_artifacts(
    filename: str, seed: int, directory: str
) -> None:
    config = load_ner_training_config(Path("configs") / filename)

    assert config.checkpoint == "./models/ner/clean/checkpoint-1045"
    assert config.seed == seed
    assert config.training["learning_rate"] == 0.00001
    assert config.training["epochs"] == 3
    assert config.training["train_batch_size"] == 8
    assert config.training["gradient_accumulation_steps"] == 2
    assert config.values["data"] == {
        "augmentation": True,
        "augmented_train_filename": "train_speech_augmented_v2.jsonl",
        "augmentation_statistics_filename": "speech_augmentation_v2_statistics.json",
    }
    assert config.output_directory.name == directory


@pytest.mark.parametrize(
    ("filename", "seed", "directory"),
    (
        (
            "ner_speech_continuation_v3_seed1.yaml",
            20260809,
            "speech-continuation-20260807-v3-seed1",
        ),
        (
            "ner_speech_continuation_v3_seed2.yaml",
            20260810,
            "speech-continuation-20260807-v3-seed2",
        ),
    ),
)
def test_v3_configs_continue_from_promoted_final_model(
    filename: str, seed: int, directory: str
) -> None:
    config = load_ner_training_config(Path("configs") / filename)

    assert config.checkpoint == "./models/ner/final"
    assert config.seed == seed
    assert config.training["learning_rate"] == 0.000005
    assert config.training["epochs"] == 2
    assert config.training["save_only_model"] is True
    assert config.values["data"]["augmented_train_filename"] == (
        "train_speech_augmented_v2.jsonl"
    )
    assert config.output_directory.name == directory


@pytest.mark.parametrize(
    ("filename", "seed", "directory"),
    (
        (
            "ner_speech_continuation_v4_seed1.yaml",
            20260811,
            "speech-continuation-20260807-v4-seed1",
        ),
        (
            "ner_speech_continuation_v4_seed2.yaml",
            20260812,
            "speech-continuation-20260807-v4-seed2",
        ),
    ),
)
def test_v4_configs_use_dense_year_curriculum_from_promoted_final(
    filename: str, seed: int, directory: str
) -> None:
    config = load_ner_training_config(Path("configs") / filename)

    assert config.checkpoint == "./models/ner/final"
    assert config.seed == seed
    assert config.training["save_only_model"] is True
    assert config.values["data"]["augmented_train_filename"] == (
        "train_speech_augmented_v3.jsonl"
    )
    assert config.values["data"]["augmentation_statistics_filename"] == (
        "speech_augmentation_v3_statistics.json"
    )
    assert config.output_directory.name == directory
