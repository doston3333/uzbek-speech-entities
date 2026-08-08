"""Train the clean Uzbek NER baseline with resumable Hugging Face checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from uzbek_speech_entities.config import project_root
from uzbek_speech_entities.ner.labels import CANONICAL_BIO_LABELS, build_label_maps
from uzbek_speech_entities.ner.training_config import NerTrainingConfig, load_ner_training_config
from uzbek_speech_entities.ner.training_data import (
    load_prepared_datasets,
    tokenize_prepared_datasets,
)
from uzbek_speech_entities.ner.training_metrics import trainer_compute_metrics
from uzbek_speech_entities.ner.training_runtime import reject_mps_fallback, select_device

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = project_root() / "data" / "processed" / "ner"
DEFAULT_CACHE_DIR = project_root() / "models" / "cache"


def parse_args() -> argparse.Namespace:
    """Parse training arguments without importing ML packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ner_clean.yaml"))
    parser.add_argument("--resume", type=Path, help="Checkpoint directory to resume exactly.")
    return parser.parse_args()


def set_all_seeds(seed: int, numpy: Any, torch: Any, transformers: Any) -> None:
    """Seed Python, NumPy, PyTorch, and the Transformers Trainer helpers."""
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    transformers.set_seed(seed)


def _assert_output_directory(output_directory: Path, resume: Path | None) -> None:
    if resume is None and output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite existing run {output_directory}; pass --resume to continue it"
        )
    if resume is not None and not resume.is_dir():
        raise FileNotFoundError(f"resume checkpoint is not a directory: {resume}")


def _package_versions() -> dict[str, str]:
    packages = (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "numpy",
        "PyYAML",
        "safetensors",
        "scikit-learn",
        "seqeval",
        "tokenizers",
        "torch",
        "transformers",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepared_dataset_statistics(datasets: Any) -> dict[str, dict[str, int]]:
    """Count source examples and words before subword tokenization."""
    statistics: dict[str, dict[str, int]] = {}
    for split_name, split in datasets.items():
        words = 0
        entity_words = 0
        for record in split:
            tags = record["ner_tags"]
            words += len(tags)
            entity_words += sum(tag != "O" for tag in tags)
        statistics[split_name] = {
            "entity_words": entity_words,
            "examples": len(split),
            "words": words,
        }
    return statistics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_provenance(
    augmentation: bool = False,
    augmented_train_filename: str = "train_augmented.jsonl",
    augmentation_statistics_filename: str = "augmentation_statistics.json",
) -> dict[str, Any]:
    statistics_path = DEFAULT_DATA_DIR / "statistics.json"
    if not statistics_path.is_file():
        raise FileNotFoundError(f"prepared dataset statistics are missing: {statistics_path}")
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    required_keys = ("dataset", "revision", "source_sha256", "seed")
    missing = [key for key in required_keys if key not in statistics]
    if missing:
        raise ValueError(f"prepared dataset statistics are missing provenance: {missing!r}")
    provenance = {key: statistics[key] for key in required_keys}
    if augmentation:
        augmentation_path = DEFAULT_DATA_DIR / augmentation_statistics_filename
        augmented_train_path = DEFAULT_DATA_DIR / augmented_train_filename
        if not augmentation_path.is_file() or not augmented_train_path.is_file():
            raise FileNotFoundError(
                "augmented dataset provenance requires configured statistics and training JSONL"
            )
        augmentation_statistics = json.loads(augmentation_path.read_text(encoding="utf-8"))
        required_augmentation_keys = (
            "allowed_transformations",
            "augmented_record_count",
            "output_record_count",
            "seed",
            "selected_source_ids_sha256",
            "source_record_count",
            "transformation_counts",
        )
        missing_augmentation = [
            key for key in required_augmentation_keys if key not in augmentation_statistics
        ]
        if missing_augmentation:
            raise ValueError(
                "augmentation statistics are missing provenance: "
                f"{missing_augmentation!r}"
            )
        provenance["augmentation"] = {
            **augmentation_statistics,
            "train_augmented_sha256": _sha256(augmented_train_path),
        }
    return provenance


def _save_run_artifacts(
    config: NerTrainingConfig,
    tokenizer: Any,
    trainer: Any,
    label2id: dict[str, int],
    id2label: dict[int, str],
    device: Any,
    dataset_statistics: dict[str, dict[str, int]],
    train_metrics: dict[str, Any],
    eval_metrics: dict[str, Any],
    duration_seconds: float,
    resume: Path | None,
    model_metadata: dict[str, Any],
) -> None:
    """Save small auditable run metadata without duplicating a model checkpoint."""
    output_directory = config.output_directory
    tokenizer.save_pretrained(output_directory)
    (output_directory / "config_snapshot.yaml").write_text(
        config.path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write_json(
        output_directory / "labels.json",
        {
            "id2label": {str(identifier): label for identifier, label in sorted(id2label.items())},
            "label2id": label2id,
            "labels": list(CANONICAL_BIO_LABELS),
        },
    )
    _write_json(output_directory / "package_versions.json", _package_versions())
    augmentation_enabled = config.values["data"]["augmentation"]
    data = config.values["data"]
    provenance = _dataset_provenance(
        augmentation_enabled,
        data.get("augmented_train_filename", "train_augmented.jsonl"),
        data.get("augmentation_statistics_filename", "augmentation_statistics.json"),
    )
    _write_json(
        output_directory / "dataset_statistics.json",
        {"provenance": provenance, "splits": dataset_statistics},
    )
    if augmentation_enabled:
        _write_json(output_directory / "augmentation_statistics.json", provenance["augmentation"])
    _write_json(output_directory / "train_metrics.json", train_metrics)
    _write_json(output_directory / "eval_metrics.json", eval_metrics)
    best_checkpoint = trainer.state.best_model_checkpoint
    if best_checkpoint is None:
        raise RuntimeError("Trainer did not record a best checkpoint")
    best_path = Path(best_checkpoint).resolve()
    try:
        relative_best_path = best_path.relative_to(output_directory.resolve())
    except ValueError as error:
        raise RuntimeError("best checkpoint is outside the configured output directory") from error
    _write_json(
        output_directory / "best_checkpoint.json",
        {
            "checkpoint": relative_best_path.as_posix(),
            "metric_for_best_model": "overall_f1",
            "metric_value": trainer.state.best_metric,
        },
    )
    _write_json(
        output_directory / "run_metadata.json",
        {
            "augmentation": augmentation_enabled,
            "best_checkpoint": relative_best_path.as_posix(),
            "device": str(device),
            "duration_seconds": duration_seconds,
            "model": model_metadata,
            "resume_from_checkpoint": str(resume) if resume else None,
        },
    )


def run_training(config: NerTrainingConfig, resume: Path | None = None) -> dict[str, Any]:
    """Run one configured training experiment. Heavy imports stay inside this path."""
    _assert_output_directory(config.output_directory, resume)
    reject_mps_fallback()
    import numpy
    import torch
    import transformers
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    device = select_device(torch)
    LOGGER.info("Selected training device: %s", device)
    set_all_seeds(config.seed, numpy, torch, transformers)
    label2id, id2label = build_label_maps()
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        config.checkpoint,
        cache_dir=str(DEFAULT_CACHE_DIR),
        fix_mistral_regex=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("NER label alignment requires a fast tokenizer with word IDs")
    model = AutoModelForTokenClassification.from_pretrained(
        config.checkpoint,
        cache_dir=str(DEFAULT_CACHE_DIR),
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    model_metadata = {
        "model_type": getattr(model.config, "model_type", None),
        "requested_checkpoint": config.checkpoint,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
    }
    prepared_datasets = load_prepared_datasets(
        DEFAULT_DATA_DIR,
        augmentation=config.values["data"]["augmentation"],
        augmented_train_filename=config.values["data"].get(
            "augmented_train_filename", "train_augmented.jsonl"
        ),
    )
    dataset_statistics = _prepared_dataset_statistics(prepared_datasets)
    tokenized_datasets, truncation_statistics = tokenize_prepared_datasets(
        prepared_datasets, tokenizer, label2id, config.max_length
    )
    for split_name, statistics in dataset_statistics.items():
        statistics.update(truncation_statistics[split_name])
    training = config.training
    arguments = TrainingArguments(
        output_dir=str(config.output_directory),
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["train_batch_size"],
        per_device_eval_batch_size=training["eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_ratio=training["warmup_ratio"],
        eval_strategy=training["eval_strategy"],
        save_strategy=training["save_strategy"],
        load_best_model_at_end=training["load_best_model_at_end"],
        metric_for_best_model=training["metric_for_best_model"],
        greater_is_better=training["greater_is_better"],
        fp16=training["fp16"],
        bf16=training["bf16"],
        dataloader_num_workers=training["dataloader_num_workers"],
        dataloader_pin_memory=False,
        report_to=[],
        save_safetensors=True,
        save_total_limit=2,
        save_only_model=training.get("save_only_model", False),
        seed=config.seed,
        data_seed=config.seed,
        use_cpu=device.type == "cpu",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=trainer_compute_metrics(id2label),
    )
    if trainer.args.device.type != device.type:
        raise RuntimeError(
            f"Trainer selected {trainer.args.device}, expected requested device {device}"
        )
    started_at = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    duration_seconds = time.monotonic() - started_at
    eval_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    train_metrics = dict(train_result.metrics)
    _save_run_artifacts(
        config,
        tokenizer,
        trainer,
        label2id,
        id2label,
        device,
        dataset_statistics,
        train_metrics,
        eval_metrics,
        duration_seconds,
        resume,
        model_metadata,
    )
    return {
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "device": str(device),
        "duration_seconds": duration_seconds,
        "eval_metrics": eval_metrics,
    }


def main() -> None:
    """Run training from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    config = load_ner_training_config(args.config)
    result = run_training(config, args.resume.resolve() if args.resume else None)
    LOGGER.info(
        "Finished NER training on %s in %.3fs; best checkpoint: %s",
        result["device"],
        result["duration_seconds"],
        result["best_checkpoint"],
    )


if __name__ == "__main__":
    main()
