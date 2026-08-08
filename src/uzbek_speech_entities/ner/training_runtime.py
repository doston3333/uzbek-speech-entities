"""Small runtime-selection helpers shared by the Phase 3 CLI scripts."""

from __future__ import annotations

import os
from typing import Any


def reject_mps_fallback() -> None:
    """Ensure MPS work cannot silently move unsupported operations to CPU."""
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is forbidden for reproducible NER runs")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"


def select_device(torch: Any) -> Any:
    """Prefer CUDA, then MPS, and select CPU explicitly as a final fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
