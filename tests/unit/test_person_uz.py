from __future__ import annotations

import pytest

from uzbek_speech_entities.ner.offset_tokens import tokenize_words
from uzbek_speech_entities.ner.rules.person import (
    person_introduction_candidates,
    person_relation_candidates,
)


def test_person_introduction_stops_at_filler_and_rejects_negation() -> None:
    text = "mening ismim rajabov doston am ismim aniq emas"
    candidates = person_introduction_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == ["rajabov doston"]


@pytest.mark.parametrize(
    "text",
    [
        "mening ismim nima",
        "mening ismim kim",
        "mening ismim emas",
        "mening ismim yoʻq",
        "mening ismim doston boraman",
        "mening ismim doston boʻlaman",
    ],
)
def test_person_introduction_rejects_questions_and_stops_at_verbs(text: str) -> None:
    candidates = person_introduction_candidates(tokenize_words(text))
    if "doston" in text:
        assert [text[item.start : item.end] for item in candidates] == ["doston"]
    else:
        assert candidates == ()


def test_person_introduction_survives_greeting_and_skips_fillers() -> None:
    text = "assalomu alaykum mening ismim am doston"
    candidates = person_introduction_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == ["doston"]


def test_person_introduction_stops_before_any_temporal_anchor() -> None:
    text = "mening ismim doston avgust kuni"
    candidates = person_introduction_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == ["doston"]


def test_person_relation_requires_name_or_surname_and_social_verb() -> None:
    text = "biz sardor bilan birga ashxobod parkida koʻrishamiz"
    candidates = person_relation_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == ["sardor"]


@pytest.mark.parametrize(
    "verb",
    [
        "koʻrishamiz",
        "uchrashamiz",
        "gaplashamiz",
        "gaplashdim",
        "boramiz",
        "keldik",
        "keladi",
        "kutamiz",
    ],
)
def test_person_relation_supports_required_social_verbs(verb: str) -> None:
    text = f"karimov bilan {verb}"
    candidates = person_relation_candidates(tokenize_words(text))
    assert [text[item.start : item.end] for item in candidates] == ["karimov"]
