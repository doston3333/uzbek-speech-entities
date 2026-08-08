"""Small immutable package-resource loaders used by speech rescue rules."""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from ..offset_tokens import comparison_key


@cache
def load_resource(name: str) -> dict[str, tuple[str, ...]]:
    """Load one bundled JSON object and freeze its string lists into tuples."""
    loaded: Any = json.loads(
        files("uzbek_speech_entities.ner.resources").joinpath(name).read_text(encoding="utf-8")
    )
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str)
        and isinstance(value, list)
        and all(isinstance(item, str) for item in value)
        for key, value in loaded.items()
    ):
        raise ValueError(f"invalid speech NER resource: {name}")
    return {key: tuple(comparison_key(item) for item in value) for key, value in loaded.items()}


@cache
def load_name_lexicon() -> frozenset[str]:
    """Read accepted lexicon entries from the counted builder output."""
    loaded: Any = json.loads(
        files("uzbek_speech_entities.ner.resources")
        .joinpath("name_lexicon.json")
        .read_text(encoding="utf-8")
    )
    names = loaded.get("names") if isinstance(loaded, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return frozenset(comparison_key(name) for name in names)
    if isinstance(names, dict) and all(isinstance(name, str) for name in names):
        return frozenset(comparison_key(name) for name in names)
    raise ValueError("invalid speech NER name lexicon")


@cache
def load_person_names() -> tuple[tuple[str, str], ...]:
    """Load reviewed single-token person names as comparison/canonical pairs."""
    loaded: Any = json.loads(
        files("uzbek_speech_entities.ner.resources")
        .joinpath("uzbek_person_names.json")
        .read_text(encoding="utf-8")
    )
    names = loaded.get("names") if isinstance(loaded, dict) else None
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError("invalid reviewed Uzbek person names")
    entries = tuple((comparison_key(name), name) for name in names)
    if len({key for key, _ in entries}) != len(entries):
        raise ValueError("duplicate reviewed Uzbek person name")
    return entries


@cache
def load_person_phrases() -> tuple[tuple[tuple[str, ...], str], ...]:
    """Load exact multi-token person phrases and their reviewed display forms."""
    loaded: Any = json.loads(
        files("uzbek_speech_entities.ner.resources")
        .joinpath("uzbek_person_phrases.json")
        .read_text(encoding="utf-8")
    )
    phrases = loaded.get("phrases") if isinstance(loaded, dict) else None
    if not isinstance(phrases, list):
        raise ValueError("invalid reviewed Uzbek person phrases")
    entries: list[tuple[tuple[str, ...], str]] = []
    for item in phrases:
        if not isinstance(item, dict):
            raise ValueError("invalid reviewed Uzbek person phrase")
        phrase, canonical = item.get("phrase"), item.get("canonical")
        if (
            not isinstance(phrase, str)
            or not isinstance(canonical, str)
            or not phrase
            or not canonical
        ):
            raise ValueError("invalid reviewed Uzbek person phrase")
        keys = tuple(comparison_key(part) for part in phrase.split())
        if not 2 <= len(keys) <= 4 or any(not key for key in keys):
            raise ValueError("person phrase must contain two to four words")
        entries.append((keys, canonical))
    if len({keys for keys, _ in entries}) != len(entries):
        raise ValueError("duplicate reviewed Uzbek person phrase")
    return tuple(entries)
