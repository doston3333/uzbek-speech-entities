"""Offline-safe prepared-dataset loading and tokenization for NER training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .alignment import TruncationStats, align_word_labels, truncation_stats
from .labels import BIOValidationError, validate_bio_record

PREPARED_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class TokenizedBatch:
    """Model inputs and transparent truncation accounting from one batch."""

    model_inputs: dict[str, Any]
    truncation: TruncationStats


def select_prepared_data_files(
    data_dir: Path,
    augmentation: bool = False,
    splits: Sequence[str] = PREPARED_SPLITS,
    *,
    augmented_train_filename: str = "train_augmented.jsonl",
) -> dict[str, Path]:
    """Return split paths, replacing only training data when augmentation is enabled."""
    requested = tuple(splits)
    unknown_splits = sorted(set(requested) - set(PREPARED_SPLITS))
    if unknown_splits:
        raise ValueError(f"unknown prepared split(s): {unknown_splits!r}")
    if not isinstance(augmentation, bool):
        raise ValueError("augmentation must be boolean")
    if (
        Path(augmented_train_filename).name != augmented_train_filename
        or not augmented_train_filename
    ):
        raise ValueError("augmented_train_filename must be a nonempty basename")
    return {
        name: data_dir
        / (augmented_train_filename if name == "train" and augmentation else f"{name}.jsonl")
        for name in requested
    }


def tokenize_prepared_batch(
    tokenizer: Any,
    batch: Mapping[str, Sequence[Sequence[str]]],
    label2id: Mapping[str, int],
    max_length: int,
) -> TokenizedBatch:
    """Tokenize split words and align their strict BIO labels without imports.

    The function deliberately has no Transformers or Datasets import so unit
    tests can use a tiny fake tokenizer entirely offline.
    """
    token_lists = batch.get("tokens")
    tag_lists = batch.get("ner_tags")
    if token_lists is None or tag_lists is None or len(token_lists) != len(tag_lists):
        raise ValueError("batch must contain equally sized tokens and ner_tags columns")
    encoded = tokenizer(
        token_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    labels: list[list[int]] = []
    aggregate = TruncationStats()
    for batch_index, ner_tags in enumerate(tag_lists):
        word_ids = encoded.word_ids(batch_index=batch_index)
        labels.append(align_word_labels(word_ids, ner_tags, label2id))
        current = truncation_stats(word_ids, ner_tags)
        aggregate = TruncationStats(
            examples=aggregate.examples + current.examples,
            words=aggregate.words + current.words,
            entity_words=aggregate.entity_words + current.entity_words,
        )
    model_inputs = dict(encoded)
    model_inputs["labels"] = labels
    return TokenizedBatch(model_inputs=model_inputs, truncation=aggregate)


def load_prepared_datasets(
    data_dir: Path,
    splits: Sequence[str] = PREPARED_SPLITS,
    *,
    augmentation: bool = False,
    augmented_train_filename: str = "train_augmented.jsonl",
) -> Any:
    """Load and strictly validate prepared JSONL splits using Hugging Face Datasets."""
    selected_files = select_prepared_data_files(
        data_dir, augmentation, splits, augmented_train_filename=augmented_train_filename
    )
    data_files = {name: str(path) for name, path in selected_files.items()}
    missing = [path for path in data_files.values() if not Path(path).is_file()]
    if missing:
        augmented_train = selected_files.get("train")
        if augmentation and augmented_train is not None and not augmented_train.is_file():
            raise FileNotFoundError(
                f"augmented training JSONL is required but missing: {augmented_train}"
            )
        raise FileNotFoundError(f"missing prepared JSONL file(s): {missing!r}")

    from datasets import load_dataset  # type: ignore[import-untyped]

    datasets = load_dataset("json", data_files=data_files)
    for split_name, split in datasets.items():
        for record in split:
            try:
                validate_bio_record(record)
            except BIOValidationError as error:
                record_id = record.get("id", "<missing>")
                raise BIOValidationError(
                    f"invalid {split_name} record {record_id!r}: {error}"
                ) from error
    return datasets


def tokenize_prepared_datasets(
    datasets: Any,
    tokenizer: Any,
    label2id: Mapping[str, int],
    max_length: int,
) -> tuple[Any, dict[str, dict[str, int]]]:
    """Tokenize prepared splits, drop source columns, and return truncation stats."""
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    tokenized_splits: dict[str, Any] = {}
    statistics: dict[str, dict[str, int]] = {}
    for split_name, split in datasets.items():
        aggregate = TruncationStats()

        def tokenize(batch: Mapping[str, Sequence[Sequence[str]]]) -> dict[str, Any]:
            nonlocal aggregate
            result = tokenize_prepared_batch(tokenizer, batch, label2id, max_length)
            aggregate = TruncationStats(
                examples=aggregate.examples + result.truncation.examples,
                words=aggregate.words + result.truncation.words,
                entity_words=aggregate.entity_words + result.truncation.entity_words,
            )
            return result.model_inputs

        tokenized_splits[split_name] = split.map(
            tokenize,
            batched=True,
            load_from_cache_file=False,
            remove_columns=split.column_names,
            desc=f"Tokenizing {split_name}",
        )
        statistics[split_name] = aggregate.as_dict()

    from datasets import DatasetDict

    return DatasetDict(tokenized_splits), statistics
