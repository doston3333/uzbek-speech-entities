from __future__ import annotations

import pytest

from uzbek_speech_entities.ner.offset_tokens import tokenize_words
from uzbek_speech_entities.ner.rules.temporal import temporal_candidates


def test_temporal_grammar_requires_a_hard_anchor() -> None:
    text = "oltinchi yu avgust kuni soati uchida uchta kitob soat sotib oldim oʻn sakkiz yoshdaman"
    candidates = temporal_candidates(tokenize_words(text))
    assert [(item.start, item.end, text[item.start : item.end]) for item in candidates] == [
        (0, 36, "oltinchi yu avgust kuni soati uchida")
    ]


def test_temporal_grammar_handles_numeric_date_and_time() -> None:
    text = "12/08/2026 soat 3:00"
    candidates = temporal_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == ["12/08/2026 soat 3:00"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("oltinchi avgust kuni", "oltinchi avgust kuni"),
        ("oltinchi yu avgust kuni soati uchida", "oltinchi yu avgust kuni soati uchida"),
        ("avgustda", "avgustda"),
        ("soat uchda", "soat uchda"),
        ("soati oʻn beshda", "soati oʻn beshda"),
        ("ertaga soat toʻrtda", "ertaga soat toʻrtda"),
        ("kelasi dushanba", "kelasi dushanba"),
        ("keyingi hafta", "keyingi hafta"),
        ("oʻtgan oy", "oʻtgan oy"),
        ("shu hafta", "shu hafta"),
        ("sentabrda", "sentabrda"),
        ("sentyabrda", "sentyabrda"),
        ("oktabrda", "oktabrda"),
        ("oktyabrda", "oktyabrda"),
        ("soat yarim", "soat yarim"),
        ("soat beshgacha", "soat beshgacha"),
        ("2026-yil 5-avgust kuni", "2026-yil 5-avgust kuni"),
        ("oʻn oltinchi avgust", "oʻn oltinchi avgust"),
    ],
)
def test_temporal_grammar_covers_required_uzbek_forms(text: str, expected: str) -> None:
    candidates = temporal_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == [expected]


@pytest.mark.parametrize(
    "text", ["oʻn sakkiz yoshdaman", "uchta kitob", "soat sotib oldim", "mayli"]
)
def test_temporal_grammar_rejects_hard_negatives(text: str) -> None:
    assert temporal_candidates(tokenize_words(text)) == ()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "oltinchi avgust ikki ming yigirma oltinchi yil",
            "oltinchi avgust ikki ming yigirma oltinchi yil",
        ),
        (
            "oltinchi avgust ikki ming yirma oltinchi yil",
            "oltinchi avgust ikki ming yirma oltinchi yil",
        ),
        (
            "bir ming toʻqqiz yuz toʻqson sakkizinchi yili",
            "bir ming toʻqqiz yuz toʻqson sakkizinchi yili",
        ),
        (
            "bir ikki ming yigirma oltinchi yil",
            "ikki ming yigirma oltinchi yil",
        ),
    ],
)
def test_temporal_grammar_merges_hard_anchored_uzbek_years(text: str, expected: str) -> None:
    candidates = temporal_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == [expected]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("oltinchi avgust ikki ming tasodifiy soʻzlar yil", ["oltinchi avgust"]),
        ("ikki ming soʻm berdim", []),
        ("oltinchi avgust ikki ming va besh yilni", ["oltinchi avgust"]),
    ],
)
def test_temporal_grammar_rejects_unanchored_or_invalid_year_sequences(
    text: str, expected: list[str]
) -> None:
    candidates = temporal_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == expected
