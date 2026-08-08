from __future__ import annotations

from uzbek_speech_entities.ner.span_resolver import Candidate, resolve_candidates


def test_resolver_uses_fixed_priority_and_vetoes_greeting_person() -> None:
    text = "assalomu alaykum doston"
    entities = resolve_candidates(
        text,
        (
            Candidate("PER", 0, 16, "clean_model", 0.99),
            Candidate("PER", 17, 23, "person_relation", evidence=("social",)),
        ),
    )
    assert [(entity.text, entity.source, entity.score) for entity in entities] == [
        ("doston", "person_relation", None)
    ]


def test_resolver_prefers_temporal_rule_over_overlapping_model_span() -> None:
    text = "avgust kuni"
    entities = resolve_candidates(
        text,
        (
            Candidate("DATE", 0, 6, "clean_model", 0.9),
            Candidate("DATE", 0, len(text), "temporal_grammar", evidence=("calendar",)),
        ),
    )
    assert [(entity.text, entity.source) for entity in entities] == [
        ("avgust kuni", "temporal_grammar")
    ]


def test_resolver_uses_per_label_caps_and_vetoes_all_greetings() -> None:
    for greeting in ("assalomu", "assalomu alaykum", "salom", "xayr", "rahmat"):
        assert (
            resolve_candidates(
                greeting,
                (Candidate("PER", 0, len(greeting), "clean_model", 0.9),),
            )
            == ()
        )
    long_date = "bir ikki uch toʻrt besh olti yetti sakkiz toʻqqiz oʻn oʻn bir"
    entity = resolve_candidates(
        long_date,
        (Candidate("DATE", 0, len(long_date), "temporal_grammar"),),
    )
    assert len(entity) == 1


def test_resolver_deduplicates_exact_label_span_before_overlap_resolution() -> None:
    text = "bugun"
    entities = resolve_candidates(
        text,
        (
            Candidate("DATE", 0, len(text), "clean_model", 0.99),
            Candidate("DATE", 0, len(text), "temporal_grammar", 0.1),
        ),
    )
    assert [(entity.text, entity.source) for entity in entities] == [("bugun", "temporal_grammar")]
