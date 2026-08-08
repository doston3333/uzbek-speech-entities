from __future__ import annotations

from training.build_name_lexicon import build_name_lexicon


def test_name_lexicon_builder_is_normalized_deterministic_and_supports_denylist() -> None:
    records = [
        {"tokens": ["Sardor", "kitob"], "ner_tags": ["B-PER", "O"]},
        {"tokens": ["sardor", "Sardor"], "ner_tags": ["O", "B-PER"]},
        {"tokens": ["Akmal"], "ner_tags": ["B-PER"]},
    ]
    assert build_name_lexicon(records) == {
        "names": {
            "akmal": {"per_count": 1, "total_count": 1, "per_ratio": 1.0},
            "sardor": {"per_count": 2, "total_count": 3, "per_ratio": 0.666667},
        }
    }
    assert build_name_lexicon(records, frozenset({"akmal"})) == {
        "names": {"sardor": {"per_count": 2, "total_count": 3, "per_ratio": 0.666667}}
    }
