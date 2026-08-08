from __future__ import annotations

import pytest

from uzbek_speech_entities.normalization import (
    normalize_evaluation,
    normalize_for_evaluation,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", ""),
        ("AKMAL KARIMOV", "akmal karimov"),
        ("Akmal, Karimov!", "akmal karimov"),
        ("O'zbekiston", "oʻzbekiston"),
        ("G’ijduvon", "gʻijduvon"),
        ("2026-yil 5-avgust.", "2026 yil 5 avgust"),
        ("soat 15:30 da", "soat 15 30 da"),
        ("  bir\tikki\nuch  ", "bir ikki uch"),
        ("Café", "café"),
        ("'Salom', dedi u.", "salom dedi u"),
        ("ertaga dushanba kuni", "ertaga dushanba kuni"),
        ("Toshkent 😊", "toshkent 😊"),
    ],
)
def test_evaluation_normalization(text: str, expected: str) -> None:
    assert normalize_evaluation(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Akmal, Karimov!",
        "O'zbekiston  Respublikasi",
        "2026 - yil 5 - avgust",
        "soat 15 : 30 da",
    ],
)
def test_evaluation_normalization_is_idempotent(text: str) -> None:
    normalized = normalize_evaluation(text)

    assert normalize_evaluation(normalized) == normalized


def test_evaluation_alias_matches_primary_interface() -> None:
    text = "O'zbekiston, Toshkent!"

    assert normalize_for_evaluation(text) == normalize_evaluation(text)
