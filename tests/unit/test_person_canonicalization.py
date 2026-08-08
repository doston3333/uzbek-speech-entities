from __future__ import annotations

from uzbek_speech_entities.ner.schemas import Entity
from uzbek_speech_entities.ner.spans import validate_entity_spans
from uzbek_speech_entities.ner.speech_extractor import SpeechNERRescue


def test_introduction_person_keeps_surface_span_and_adds_canonical_name() -> None:
    text = "mening ismim doskon"
    entities = SpeechNERRescue().extract(text, ())
    assert [(entity.text, entity.label, entity.canonical_text) for entity in entities] == [
        ("doskon", "PER", "Doston")
    ]
    assert entities[0].canonical_source == "name_lexicon_edit_distance"
    assert text[entities[0].start : entities[0].end] == "doskon"


def test_unrelated_near_name_does_not_create_a_person() -> None:
    assert SpeechNERRescue().extract("doston janri haqida gaplashamiz", ()) == ()


def test_exact_person_phrase_and_organization_disambiguation() -> None:
    rescue = SpeechNERRescue()
    person = rescue.extract("alisher navoiy buyuk shoir", ())
    assert [
        (entity.text, entity.label, entity.source, entity.canonical_text) for entity in person
    ] == [
        ("alisher navoiy", "PER", "person_gazetteer", "Alisher Navoiy")
    ]
    organization_text = "alisher navoiy nomidagi universitetda"
    organization = rescue.extract(organization_text, ())
    assert [(entity.text, entity.label, entity.source) for entity in organization] == [
        (organization_text, "ORG", "gazetteer_boundary_expansion")
    ]
    theater_text = "alisher navoiy teatri"
    theater = rescue.extract(theater_text, ())
    assert [(entity.text, entity.label, entity.source) for entity in theater] == [
        (theater_text, "ORG", "gazetteer_boundary_expansion")
    ]


def test_person_phrase_does_not_cross_punctuation() -> None:
    assert SpeechNERRescue().extract("alisher, navoiy buyuk shoir", ()) == ()


def test_location_model_anchor_remains_location_without_single_token_phrase_rule() -> None:
    text = "navoiy viloyatida"
    entities = SpeechNERRescue().extract(
        text,
        (Entity(text="navoiy", label="LOC", start=0, end=6, score=0.9),),
    )
    assert [(entity.text, entity.label, entity.source) for entity in entities] == [
        (text, "LOC", "model_boundary_expansion")
    ]


def test_repeated_relative_dates_and_rule_model_duplicate_are_exact_and_ordered() -> None:
    text = "bugun bugun bugun"
    model = (Entity(text="bugun", label="DATE", start=0, end=5, score=0.99),)
    entities = SpeechNERRescue().extract(text, model)
    assert [(entity.start, entity.end, entity.text, entity.source) for entity in entities] == [
        (0, 5, "bugun", "temporal_grammar"),
        (6, 11, "bugun", "temporal_grammar"),
        (12, 17, "bugun", "temporal_grammar"),
    ]
    validate_entity_spans(text, entities)
