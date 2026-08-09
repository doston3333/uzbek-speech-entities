"""Pinned Uzbek NER dataset metadata and integrity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

DATASET_ID: Final[str] = "uznlp-uz/uzbek_NER"
DATASET_REVISION: Final[str] = "4825ce29a1372bd78cb1cbb73693f16ec6f8328d"
DATASET_FILENAME: Final[str] = "Uzbek_NER_Gold.tsv"
DATASET_URL: Final[str] = (
    "https://huggingface.co/datasets/uznlp-uz/uzbek_NER/resolve/"
    f"{DATASET_REVISION}/{DATASET_FILENAME}?download=true"
)
DATASET_SHA256: Final[str] = "45acfdc7fabb8668b383b1fee31d2541c767aa8d05e9a0b1e70475f1b424eac8"
SOURCE_COLUMNS: Final[tuple[str, ...]] = (
    "Sentence",
    "TokenOrder",
    "Token",
    "NER_Tag",
    "pos",
)


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
