"""Evaluate a local Phase 3 NER checkpoint on one prepared split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from uzbek_speech_entities.config import resolve_project_path
from uzbek_speech_entities.ner.training_config import load_ner_training_config
from uzbek_speech_entities.ner.training_data import (
    load_prepared_datasets,
    tokenize_prepared_datasets,
)
from uzbek_speech_entities.ner.training_metrics import compute_ner_metrics
from uzbek_speech_entities.ner.training_runtime import reject_mps_fallback, select_device

LOGGER = logging.getLogger(__name__)
DEFAULT_POINTER = Path("models/ner/clean/best_checkpoint.json")
DEFAULT_DATA_DIR = resolve_project_path("data/processed/ner")


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments without importing the model stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--config", type=Path, default=Path("configs/ner_clean.yaml"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, help="Deterministic JSON metrics destination.")
    return parser.parse_args()


def resolve_checkpoint(checkpoint: Path) -> tuple[Path, Path]:
    """Resolve a direct checkpoint, run directory, or auditable pointer JSON."""
    candidate = resolve_project_path(checkpoint)
    if (
        candidate.is_dir()
        and (candidate / "model.safetensors").is_file()
        and (candidate / "labels.json").is_file()
    ):
        # A compact inference bundle is self-contained even when it retains the
        # source run's best-checkpoint pointer for provenance.
        return candidate, candidate
    if candidate.is_dir() and (candidate / "best_checkpoint.json").is_file():
        return resolve_checkpoint(candidate / "best_checkpoint.json")
    if candidate.is_file():
        pointer = json.loads(candidate.read_text(encoding="utf-8"))
        target = pointer.get("checkpoint")
        if not isinstance(target, str) or not target:
            raise ValueError(f"invalid best-checkpoint pointer: {candidate}")
        resolved = (candidate.parent / target).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"checkpoint pointer target is not a directory: {resolved}")
        return resolved, candidate.parent
    if candidate.is_dir():
        if (candidate / "labels.json").is_file():
            # A compact promoted inference directory keeps run metadata beside
            # the model instead of in the checkpoint's parent directory.
            return candidate, candidate
        artifact_root = candidate.parent
        if not (artifact_root / "labels.json").is_file():
            raise FileNotFoundError(f"labels.json not found next to checkpoint: {candidate}")
        return candidate, artifact_root
    raise FileNotFoundError(f"checkpoint or pointer does not exist: {candidate}")


def load_label_maps(artifact_root: Path) -> tuple[dict[str, int], dict[int, str]]:
    """Load the exact training label maps retained with a run."""
    saved = json.loads((artifact_root / "labels.json").read_text(encoding="utf-8"))
    label2id = saved.get("label2id")
    raw_id2label = saved.get("id2label")
    if not isinstance(label2id, dict) or not isinstance(raw_id2label, dict):
        raise ValueError(f"invalid labels artifact: {artifact_root / 'labels.json'}")
    id2label = {int(identifier): label for identifier, label in raw_id2label.items()}
    is_consistent = all(
        isinstance(label, str) and label2id.get(label) == identifier
        for identifier, label in id2label.items()
    )
    if not is_consistent:
        raise ValueError(f"inconsistent labels artifact: {artifact_root / 'labels.json'}")
    return {str(label): int(identifier) for label, identifier in label2id.items()}, id2label


def evaluate(
    checkpoint: Path,
    artifact_root: Path,
    config_path: Path,
    split_name: str,
) -> dict[str, float | int]:
    """Run local prediction and return the same entity-first training metrics."""
    reject_mps_fallback()
    import torch
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    config = load_ner_training_config(config_path)
    label2id, id2label = load_label_maps(artifact_root)
    device = select_device(torch)
    LOGGER.info("Selected evaluation device: %s", device)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        artifact_root, fix_mistral_regex=False
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("NER label alignment requires a fast tokenizer with word IDs")
    model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    prepared = load_prepared_datasets(DEFAULT_DATA_DIR, splits=(split_name,))
    tokenized, _truncation = tokenize_prepared_datasets(
        prepared, tokenizer, label2id, config.max_length
    )
    arguments = TrainingArguments(
        output_dir=str(artifact_root),
        per_device_eval_batch_size=config.training["eval_batch_size"],
        dataloader_pin_memory=False,
        report_to=[],
        use_cpu=device.type == "cpu",
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    if trainer.args.device.type != device.type:
        raise RuntimeError(
            f"Trainer selected {trainer.args.device}, expected requested device {device}"
        )
    prediction = trainer.predict(tokenized[split_name])
    return compute_ner_metrics(prediction.predictions, prediction.label_ids, id2label)


def write_metrics(path: Path, metrics: dict[str, float | int]) -> None:
    """Write deterministic metrics JSON for experiment comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Evaluate a supplied or default clean best checkpoint pointer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    checkpoint, artifact_root = resolve_checkpoint(args.checkpoint)
    metrics = evaluate(checkpoint, artifact_root, args.config, args.split)
    output = args.output or artifact_root / f"evaluation_{args.split}_metrics.json"
    write_metrics(resolve_project_path(output), metrics)
    LOGGER.info("Wrote %s metrics to %s", args.split, resolve_project_path(output))


if __name__ == "__main__":
    main()
