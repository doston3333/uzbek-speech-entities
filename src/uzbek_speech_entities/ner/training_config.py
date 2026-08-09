"""Strict, dependency-light Phase 3 NER training configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from uzbek_speech_entities.config import project_root, resolve_project_path


@dataclass(frozen=True)
class NerTrainingConfig:
    """Validated configuration values consumed directly by the Phase 3 scripts."""

    path: Path
    values: Mapping[str, Any]
    checkpoint: str
    max_length: int
    seed: int
    output_directory: Path

    @property
    def training(self) -> Mapping[str, Any]:
        return _mapping(self.values["training"], "training")


_REQUIRED_TRAINING_KEYS = frozenset(
    {
        "seed",
        "learning_rate",
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "weight_decay",
        "warmup_ratio",
        "eval_strategy",
        "save_strategy",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
        "fp16",
        "bf16",
        "dataloader_num_workers",
    }
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"missing or invalid {name} configuration section")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _basename(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{name} must be a nonempty basename")
    return value


def load_ner_training_config(path: str | Path = "configs/ner_clean.yaml") -> NerTrainingConfig:
    """Load and validate a Phase 3 YAML file without importing ML libraries."""
    import yaml

    config_path = resolve_project_path(path)
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    root = _mapping(values, "root")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    data = _mapping(root.get("data"), "data")
    output = _mapping(root.get("output"), "output")
    missing = sorted(_REQUIRED_TRAINING_KEYS - set(training))
    if missing:
        raise ValueError(f"missing training configuration key(s): {missing!r}")
    checkpoint = model.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("model.checkpoint must be a nonempty string")
    max_length = _positive_int(model.get("max_length"), "model.max_length")
    seed = _positive_int(training.get("seed"), "training.seed")
    for key in ("epochs", "train_batch_size", "eval_batch_size", "gradient_accumulation_steps"):
        _positive_int(training[key], f"training.{key}")
    if (
        not isinstance(training["dataloader_num_workers"], int)
        or training["dataloader_num_workers"] < 0
    ):
        raise ValueError("training.dataloader_num_workers must be nonnegative")
    for key in ("learning_rate", "weight_decay", "warmup_ratio"):
        if isinstance(training[key], bool) or not isinstance(training[key], int | float):
            raise ValueError(f"training.{key} must be numeric")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("training learning_rate must be positive and weight_decay nonnegative")
    if not 0 <= training["warmup_ratio"] <= 1:
        raise ValueError("training.warmup_ratio must be between zero and one")
    if training["eval_strategy"] != "epoch" or training["save_strategy"] != "epoch":
        raise ValueError("Phase 3 eval_strategy and save_strategy must both be 'epoch'")
    if training["metric_for_best_model"] != "overall_f1":
        raise ValueError("Phase 3 metric_for_best_model must be overall_f1")
    for key in ("load_best_model_at_end", "greater_is_better", "fp16", "bf16"):
        if not isinstance(training[key], bool):
            raise ValueError(f"training.{key} must be boolean")
    if "save_only_model" in training and not isinstance(training["save_only_model"], bool):
        raise ValueError("training.save_only_model must be boolean")
    if training["fp16"] and training["bf16"]:
        raise ValueError("training.fp16 and training.bf16 cannot both be true")
    if not isinstance(data.get("augmentation"), bool):
        raise ValueError("data.augmentation must be boolean")
    for key in ("augmented_train_filename", "augmentation_statistics_filename"):
        if key in data:
            _basename(data[key], f"data.{key}")
    directory = output.get("directory")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError("output.directory must be a nonempty string")
    return NerTrainingConfig(
        path=config_path,
        values=root,
        checkpoint=checkpoint,
        max_length=max_length,
        seed=seed,
        output_directory=resolve_project_path(directory, project_root()),
    )
