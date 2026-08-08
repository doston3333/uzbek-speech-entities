from __future__ import annotations

from uzbek_speech_entities.ner.offset_tokens import tokenize_words
from uzbek_speech_entities.ner.rules.boundaries import boundary_expansion_candidates
from uzbek_speech_entities.ner.schemas import Entity


def test_boundary_completion_needs_model_anchor_and_semantic_head() -> None:
    text = "yangi oʻzbekiston universitetida ashxobod parkida"
    anchor = Entity(
        text="oʻzbekiston",
        label="LOC",
        start=text.index("oʻzbekiston"),
        end=text.index("oʻzbekiston") + len("oʻzbekiston"),
        score=0.9,
    )
    candidates = boundary_expansion_candidates(tokenize_words(text), (anchor,))
    assert [(item.label, text[item.start : item.end]) for item in candidates] == [
        ("ORG", "yangi oʻzbekiston universitetida")
    ]


def test_boundary_completion_retypes_loc_anchor_and_stops_at_boundaries() -> None:
    text = "ashxobod parkida yangi universitet"
    anchor = Entity(
        text="ashxobod",
        label="ORG",
        start=0,
        end=len("ashxobod"),
        score=0.9,
    )
    candidates = boundary_expansion_candidates(tokenize_words(text), (anchor,))
    assert [(item.label, text[item.start : item.end]) for item in candidates] == [
        ("LOC", "ashxobod parkida")
    ]
    assert boundary_expansion_candidates(tokenize_words("yangi universitet"), ()) == ()


def test_boundary_completion_handles_required_inflected_heads() -> None:
    cases = [
        ("navoiy ko'chasida", "navoiy", "LOC"),
        ("toshkent shahrida", "toshkent", "LOC"),
        ("zomin qishlog'ida", "zomin", "LOC"),
        ("yoshlar agentligida", "yoshlar", "ORG"),
    ]
    for text, anchor_text, expected_label in cases:
        anchor = Entity(
            text=anchor_text,
            label="LOC",
            start=0,
            end=len(anchor_text),
            score=0.9,
        )
        candidates = boundary_expansion_candidates(tokenize_words(text), (anchor,))
        assert [(item.label, text[item.start : item.end]) for item in candidates] == [
            (expected_label, text)
        ]
