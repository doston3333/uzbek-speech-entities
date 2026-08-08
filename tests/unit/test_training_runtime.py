from __future__ import annotations

from types import SimpleNamespace

import pytest

from uzbek_speech_entities.ner.training_runtime import select_device


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")),
)
def test_select_device_prefers_cuda_then_mps_then_cpu(
    cuda_available: bool, mps_available: bool, expected: str
) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda_available),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        device=lambda name: f"device:{name}",
    )

    assert select_device(torch) == f"device:{expected}"
