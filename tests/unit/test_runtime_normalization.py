from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

import pytest

from uzbek_speech_entities.normalization import normalize_runtime


class NormalizationCase(TypedDict):
    id: str
    input: str
    expected: str


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures/normalization_cases.json"
CASES: list[NormalizationCase] = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_runtime_normalization_cases(case: NormalizationCase) -> None:
    assert normalize_runtime(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_runtime_normalization_is_idempotent(case: NormalizationCase) -> None:
    normalized = normalize_runtime(case["input"])

    assert normalize_runtime(normalized) == normalized


def test_runtime_fixture_contains_at_least_thirty_cases() -> None:
    assert len(CASES) >= 30
