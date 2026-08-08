"""Reusable Phase 8 evaluation dataset and metric helpers."""

from typing import Any

__all__ = ["APPLICATION_LABELS"]


def __getattr__(name: str) -> Any:
    """Lazily preserve the public label export without preloading CLI modules."""
    if name == "APPLICATION_LABELS":
        from .dataset import APPLICATION_LABELS

        return APPLICATION_LABELS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
