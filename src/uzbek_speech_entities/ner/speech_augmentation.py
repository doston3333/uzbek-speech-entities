"""Deterministic, train-only augmentation for speech-shaped Uzbek NER input.

This module intentionally never consumes held-out labels.  Protected JSONL
files are used only to fingerprint normalized full texts, which prevents a
synthetic sentence from leaking into evaluation material.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .labels import BIOValidationError, validate_bio_record, validate_bio_sequence

DEFAULT_SEED = 20260806
DEFAULT_PROTECTED_PATHS = (
    Path("data/processed/ner/validation.jsonl"),
    Path("data/processed/ner/test.jsonl"),
    Path("tests/fixtures/speech_ner_eval.jsonl"),
)
FILLERS = ("am", "um", "ee")
ENTITY_TYPES = frozenset({"PER", "LOC", "ORG", "TEMPORAL"})
_PUNCTUATION = frozenset('.,!?;:…()[]{}"“”‘’')
_MONTHS = (
    "yanvar",
    "fevral",
    "mart",
    "aprel",
    "may",
    "iyun",
    "iyul",
    "avgust",
    "sentabr",
    "oktabr",
    "noyabr",
    "dekabr",
)
_ORDINALS = {
    1: ("birinchi",),
    2: ("ikkinchi",),
    3: ("uchinchi",),
    4: ("toʻrtinchi",),
    5: ("beshinchi",),
    6: ("oltinchi",),
    7: ("yettinchi",),
    8: ("sakkizinchi",),
    9: ("toʻqqizinchi",),
    10: ("oʻninchi",),
    11: ("oʻn", "birinchi"),
    12: ("oʻn", "ikkinchi"),
    13: ("oʻn", "uchinchi"),
    14: ("oʻn", "toʻrtinchi"),
    15: ("oʻn", "beshinchi"),
    16: ("oʻn", "oltinchi"),
    17: ("oʻn", "yettinchi"),
    18: ("oʻn", "sakkizinchi"),
    19: ("oʻn", "toʻqqizinchi"),
    20: ("yigirmanchi",),
    21: ("yigirma", "birinchi"),
    22: ("yigirma", "ikkinchi"),
    23: ("yigirma", "uchinchi"),
    24: ("yigirma", "toʻrtinchi"),
    25: ("yigirma", "beshinchi"),
    26: ("yigirma", "oltinchi"),
    27: ("yigirma", "yettinchi"),
    28: ("yigirma", "sakkizinchi"),
    29: ("yigirma", "toʻqqizinchi"),
    30: ("oʻttizinchi",),
    31: ("oʻttiz", "birinchi"),
}
_CARDINALS = {
    0: "",
    1: "bir",
    2: "ikki",
    3: "uch",
    4: "toʻrt",
    5: "besh",
    6: "olti",
    7: "yetti",
    8: "sakkiz",
    9: "toʻqqiz",
    10: "oʻn",
    20: "yigirma",
    30: "oʻttiz",
}
_APOSTROPHE_TRANSLATION = str.maketrans({"'": "ʻ", "’": "ʻ", "‘": "ʻ", "ʼ": "ʻ", "`": "ʻ"})
# This is deliberately a fingerprint-only denylist; it is never decoded into
# training text.  The value protects the held-out 2026-08-06 OGG transcript.
KNOWN_OGG_TRANSCRIPT_FINGERPRINTS: frozenset[str] = frozenset(
    {
        # Populated from the normalized full transcript, never from entity labels.
        "74021c40cd631abaad79e31cc8ab58daa223f3d73cab1208ab173f3fd48726a0",
    }
)


def normalize_full_text(value: str | Sequence[str]) -> str:
    """Return a comparison key for full-text leakage checks."""
    text = value if isinstance(value, str) else " ".join(value)
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHE_TRANSLATION).casefold()
    return " ".join(re.sub(r"[^\wʻ]+", " ", normalized).split())


def full_text_fingerprint(value: str | Sequence[str]) -> str:
    return hashlib.sha256(normalize_full_text(value).encode("utf-8")).hexdigest()


def normalized_ngrams(
    value: str | Sequence[str], *, minimum_length: int = 3, maximum_length: int = 5
) -> frozenset[tuple[str, ...]]:
    """Return normalized token n-grams used to protect authored templates."""
    if minimum_length < 1 or maximum_length < minimum_length:
        raise ValueError("ngram bounds must be positive and ordered")
    tokens = tuple(normalize_full_text(value).split())
    return frozenset(
        tokens[start : start + length]
        for length in range(minimum_length, maximum_length + 1)
        for start in range(len(tokens) - length + 1)
    )


def _ngram_digest(ngrams: Iterable[tuple[str, ...]]) -> str:
    return hashlib.sha256(
        "\n".join(" ".join(ngram) for ngram in sorted(ngrams)).encode("utf-8")
    ).hexdigest()


def _entity_type(tag: str) -> str | None:
    return tag.split("-", 1)[1] if "-" in tag else None


def _is_viable(record: Mapping[str, Any]) -> bool:
    return any(_entity_type(tag) in ENTITY_TYPES for tag in record["ner_tags"])


def _normalize_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).translate(_APOSTROPHE_TRANSLATION).casefold()


def _tags_for(tag: str, count: int) -> list[str]:
    if tag == "O":
        return ["O"] * count
    entity_type = _entity_type(tag)
    assert entity_type is not None
    return [f"B-{entity_type}", *([f"I-{entity_type}"] * (count - 1))]


def _year_words(year: int) -> tuple[str, ...]:
    if not 2000 <= year <= 2030:
        return ()
    remainder = year - 2000
    if remainder == 0:
        return ("ikki", "ming")
    tens, units = divmod(remainder, 10)
    tail = ([] if tens == 0 else [_CARDINALS[tens * 10]]) + (
        [] if units == 0 else [_CARDINALS[units]]
    )
    return ("ikki", "ming", *tail)


def _spoken_year_variants(year: int) -> tuple[tuple[str, ...], ...]:
    """Return common 2000--2030 year forms, including the optional connector."""
    words = _year_words(year)
    if len(words) <= 2:
        return (words,)
    return (words, (*words[:2], "va", *words[2:]))


def expand_temporal_tokens(
    tokens: Sequence[str], tags: Sequence[str]
) -> tuple[list[str], list[str], int]:
    """Speak unambiguous TEMPORAL day/month and 2000--2030 year forms only."""
    validate_bio_sequence(tokens, tags)
    result_tokens: list[str] = []
    result_tags: list[str] = []
    expansions = 0
    index = 0
    while index < len(tokens):
        token, tag = tokens[index], tags[index]
        entity_type = _entity_type(tag)
        normalized = _normalize_token(token)
        replacement: tuple[str, ...] = ()
        consumed = 1
        if entity_type == "TEMPORAL" and tag.startswith("B-"):
            if "-" in normalized:
                day, separator, month = normalized.partition("-")
                if separator and day.isdigit() and int(day) in _ORDINALS and month in _MONTHS:
                    replacement = (*_ORDINALS[int(day)], month)
            if not replacement and normalized.isdigit() and int(normalized) in _ORDINALS:
                if index + 1 < len(tokens) and tags[index + 1] == "I-TEMPORAL":
                    next_token = _normalize_token(tokens[index + 1])
                    if next_token in _MONTHS:
                        replacement = (*_ORDINALS[int(normalized)], next_token)
                        consumed = 2
            if not replacement and normalized.endswith("-yil"):
                year_text = normalized.removesuffix("-yil")
                if year_text.isdigit():
                    replacement = (*_year_words(int(year_text)), "yil")
            if (
                not replacement
                and normalized.isdigit()
                and index + 1 < len(tokens)
                and tags[index + 1] == "I-TEMPORAL"
            ):
                next_token = _normalize_token(tokens[index + 1])
                if next_token in {"yil", "yili", "yilda"}:
                    words = _year_words(int(normalized))
                    if words:
                        replacement = (*words, "yil")
                        consumed = 2
        if replacement:
            result_tokens.extend(replacement)
            result_tags.extend(_tags_for(tag, len(replacement)))
            expansions += 1
            index += consumed
            continue
        result_tokens.append(normalized)
        result_tags.append(tag)
        index += 1
    validate_bio_sequence(result_tokens, result_tags)
    return result_tokens, result_tags, expansions


def speech_variant(
    record: Mapping[str, Any], seed: int = DEFAULT_SEED
) -> tuple[list[str], list[str], dict[str, int]]:
    """Produce one lower-case, punctuation-light variant without changing entities."""
    tokens = list(record["tokens"])
    tags = list(record["ner_tags"])
    validate_bio_sequence(tokens, tags)
    stripped_tokens = [
        (token, tag)
        for token, tag in zip(tokens, tags, strict=True)
        if not (tag == "O" and token and all(char in _PUNCTUATION for char in token))
    ]
    base_tokens = [token for token, _ in stripped_tokens]
    base_tags = [tag for _, tag in stripped_tokens]
    converted_tokens, converted_tags, expanded = expand_temporal_tokens(base_tokens, base_tags)
    counts = {
        "lowercase": 0,
        "o_punctuation_removed": len(tokens) - len(base_tokens),
        "temporal_spoken": expanded,
        "filler": 0,
    }
    digest = int(hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest()[:8], 16)
    safe_positions = [
        index + 1
        for index, tag in enumerate(converted_tags)
        if tag == "O" and converted_tokens[index].isalnum()
    ]
    if safe_positions:
        position = safe_positions[digest % len(safe_positions)]
        converted_tokens.insert(position, FILLERS[digest % len(FILLERS)])
        converted_tags.insert(position, "O")
        counts["filler"] = 1
    counts["lowercase"] = sum(token != _normalize_token(token) for token, _ in stripped_tokens)
    validate_bio_sequence(converted_tokens, converted_tags)
    return converted_tokens, converted_tags, counts


def _record(
    identifier: str, tokens: Sequence[str], tags: Sequence[str], template: str
) -> dict[str, Any]:
    record = {
        "id": identifier,
        "tokens": list(tokens),
        "ner_tags": list(tags),
        "source": "speech_ner_template",
        "augmentation": {"template": template},
    }
    validate_bio_record(record)
    return record


def _phrases(records: Sequence[Mapping[str, Any]], label: str, limit: int) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for record in records:
        tokens, tags = record["tokens"], record["ner_tags"]
        start = 0
        while start < len(tokens):
            if tags[start] == f"B-{label}":
                end = start + 1
                while end < len(tokens) and tags[end] == f"I-{label}":
                    end += 1
                phrase = tuple(_normalize_token(token) for token in tokens[start:end])
                if phrase and phrase not in seen and all(token.isalpha() for token in phrase):
                    found.append(phrase)
                    seen.add(phrase)
                    if len(found) == limit:
                        return found
                start = end
            else:
                start += 1
    return found


def authored_templates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return bounded train-only conversational positives and explicit negatives."""
    templates: list[dict[str, Any]] = []
    names = (
        "akmal",
        "ali",
        "aziz",
        "doston",
        "javohir",
        "muhammad",
        "nodir",
        "rajab",
        "sardor",
        "shohruh",
        "temur",
        "umid",
    )
    for index, name in enumerate(names):
        for prefix in (
            ("mening", "ismim"),
            ("mening", "ismim", "am"),
            ("salom", "ismim"),
            ("men", "bugun"),
            ("biz",),
        ):
            if prefix == ("men", "bugun"):
                tokens = [*prefix, name, "bilan", "gaplashdim"]
            elif prefix == ("biz",):
                tokens = [*prefix, name, "bilan", "uchrashamiz"]
            else:
                tokens = [*prefix, name, "men"]
            tags = ["O"] * len(tokens)
            tags[len(prefix)] = "B-PER"
            if prefix == ("men", "bugun"):
                tags[1] = "B-TEMPORAL"
            templates.append(
                _record(
                    f"speech-template-per-{index:02d}-{len(templates):03d}",
                    tokens,
                    tags,
                    "per_intro_or_relation",
                )
            )
        variant = name[:-1] + ("a" if name[-1] != "a" else "o")
        if variant != "doskon":
            templates.append(
                _record(
                    f"speech-template-per-edit-{index:02d}",
                    ["mening", "ismim", variant],
                    ["O", "O", "B-PER"],
                    "per_one_edit_intro",
                )
            )
    for day in (1, 6, 16, 21, 31):
        for month in _MONTHS:
            words = [*_ORDINALS[day], month, "kuni", "kelaman"]
            templates.append(
                _record(
                    f"speech-template-date-{day:02d}-{month}",
                    words,
                    _tags_for("B-TEMPORAL", len(words) - 1) + ["O"],
                    "spoken_calendar",
                )
            )
    for index, phrase in enumerate(
        (
            ("bugun",),
            ("ertaga",),
            ("kelasi", "seshanba"),
            ("keyingi", "hafta"),
            ("oʻtgan", "oy"),
            ("shu", "hafta"),
        )
    ):
        templates.append(
            _record(
                f"speech-template-relative-{index:02d}",
                [*phrase, "uchrashamiz"],
                _tags_for("B-TEMPORAL", len(phrase)) + ["O"],
                "relative_calendar",
            )
        )
    year_contexts = (("yil", "boshlandi"), ("yilda", "uchrashamiz"), ("yil", "reja", "tuzdik"))
    for year in range(2000, 2031):
        # Keep a deterministic subset of years unseen so this remains a
        # compositional curriculum instead of memorizing every target value.
        if (year - 2000) % 11 == 5:
            continue
        for variant_index, year_words in enumerate(_spoken_year_variants(year)):
            for context_index, context in enumerate(year_contexts):
                words = [*year_words, *context]
                entity_length = len(year_words) + 1
                templates.append(
                    _record(
                        f"speech-template-year-{year}-{variant_index}-{context_index}",
                        words,
                        _tags_for("B-TEMPORAL", entity_length)
                        + ["O"] * (len(words) - entity_length),
                        "spoken_year",
                    )
                )
    for index, phrase in enumerate(_phrases(records, "PER", 64)):
        tokens = ["bugun", *phrase, "bilan", "uchrashamiz"]
        templates.append(
            _record(
                f"speech-template-per-inventory-{index:03d}",
                tokens,
                ["B-TEMPORAL", *_tags_for("B-PER", len(phrase)), "O", "O"],
                "per_train_inventory",
            )
        )
    boundary_curricula = (
        (
            "ORG",
            "ishlayman",
            ("tashkiloti", "tashkilotida", "kompaniyasi", "kompaniyasida", "markazi", "markazida"),
        ),
        (
            "LOC",
            "boraman",
            ("shahri", "shahrida", "tumani", "tumanida", "viloyati", "viloyatida"),
        ),
    )
    for label, tail, boundary_heads in boundary_curricula:
        for phrase_index, phrase in enumerate(_phrases(records, label, 24)):
            for head_index, head in enumerate(boundary_heads):
                tokens = [*phrase, head, tail]
                templates.append(
                    _record(
                        f"speech-template-{label.lower()}-boundary-{phrase_index:03d}-{head_index:02d}",
                        tokens,
                        _tags_for(f"B-{label}", len(phrase) + 1) + ["O"],
                        "entity_boundary_curriculum",
                    )
                )
    for label, negative_heads in (
        ("ORG", ("tashkilot", "kompaniya", "markaz")),
        ("LOC", ("shahar", "tuman", "viloyat")),
    ):
        for head_index, head in enumerate(negative_heads):
            templates.append(
                _record(
                    f"speech-template-{label.lower()}-head-negative-{head_index:02d}",
                    [head, "haqida", "gaplashamiz"],
                    ["O", "O", "O"],
                    "entity_head_hard_negative",
                )
            )
    eponymic_location_heads = ("koʻchasi", "koʻchasida", "bogʻi", "bogʻida")
    for index, phrase in enumerate(_phrases(records, "PER", 24)):
        templates.append(
            _record(
                f"speech-template-eponymic-per-{index:03d}",
                ["men", *phrase, "bilan", "gaplashdim"],
                ["O", *_tags_for("B-PER", len(phrase)), "O", "O"],
                "eponymic_loc_per_contrast",
            )
        )
        for head_index, head in enumerate(eponymic_location_heads):
            templates.append(
                _record(
                    f"speech-template-eponymic-loc-{index:03d}-{head_index:02d}",
                    [*phrase, head, "boraman"],
                    _tags_for("B-LOC", len(phrase) + 1) + ["O"],
                    "eponymic_loc_per_contrast",
                )
            )
    negatives = (
        ("men", "oʻn", "sakkiz", "yoshdaman"),
        ("uchta", "kitob", "oldim"),
        ("salom", "um", "rahmat"),
        ("200yil", "kelaman"),
        ("2031-yil", "emas"),
        ("universitet", "yonida", "turaman"),
        ("park", "yaqinida", "kutaman"),
        ("bank", "ochiq", "emas"),
        ("ee", "men", "kelaman"),
        ("soat", "uchta", "kitob"),
        ("ikki", "ming", "tushunarsiz", "yil"),
        ("yigirma", "beshta", "odam"),
        ("men", "oʻttiz", "yoshdaman"),
    )
    for repeat in range(5):
        for index, negative_tokens in enumerate(negatives):
            templates.append(
                _record(
                    f"speech-template-negative-{repeat:02d}-{index:02d}",
                    negative_tokens,
                    ["O"] * len(negative_tokens),
                    "hard_negative",
                )
            )
    return templates


def _derived_id(source_id: str, ordinal: int, used: set[str]) -> str:
    candidate = f"{source_id}__speech_{ordinal:04d}"
    suffix = 2
    while candidate in used:
        candidate = f"{source_id}__speech_{ordinal:04d}_{suffix}"
        suffix += 1
    return candidate


def build_speech_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    protected_fingerprints: frozenset[str] = frozenset(),
    protected_authored_template_ngrams: frozenset[tuple[str, ...]] = frozenset(),
    denylisted_fingerprints: frozenset[str] = KNOWN_OGG_TRANSCRIPT_FINGERPRINTS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Append one variant per viable source and bounded authored templates."""
    source_records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise BIOValidationError(f"source record at index {index} is not an object")
        record = dict(item)
        validate_bio_record(record)
        identifier = record["id"]
        if identifier in identifiers:
            raise ValueError(f"duplicate source record ID: {identifier!r}")
        identifiers.add(identifier)
        source_records.append(record)
    excluded_source_ids = [
        str(record["id"])
        for record in source_records
        if full_text_fingerprint(record["tokens"]) in protected_fingerprints
    ]
    originals = [
        record
        for record in source_records
        if full_text_fingerprint(record["tokens"]) not in protected_fingerprints
    ]
    output = [dict(record) for record in originals]
    transform_counts: Counter[str] = Counter()
    source_ids: list[str] = []
    for ordinal, source in enumerate((record for record in originals if _is_viable(record)), 1):
        tokens, tags, counts = speech_variant(source, seed)
        variant = dict(source)
        variant["id"] = _derived_id(str(source["id"]), ordinal, identifiers)
        variant["tokens"], variant["ner_tags"] = tokens, tags
        variant["augmentation"] = {
            "source_id": source["id"],
            "transformations": sorted(key for key, value in counts.items() if value),
        }
        validate_bio_record(variant)
        identifiers.add(variant["id"])
        output.append(variant)
        source_ids.append(str(source["id"]))
        transform_counts.update(counts)
    template_counts: Counter[str] = Counter()
    excluded_template_count = 0
    excluded_template_ids: list[str] = []
    included_template_ngrams: set[tuple[str, ...]] = set()
    for template in authored_templates(originals):
        template_ngrams = normalized_ngrams(template["tokens"])
        if template_ngrams & protected_authored_template_ngrams:
            excluded_template_count += 1
            excluded_template_ids.append(str(template["id"]))
            continue
        if template["id"] in identifiers:
            raise ValueError(f"duplicate synthetic ID: {template['id']!r}")
        identifiers.add(template["id"])
        output.append(template)
        included_template_ngrams.update(template_ngrams)
        template_counts[str(template["augmentation"]["template"])] += 1
    included_template_ngram_overlap = included_template_ngrams & protected_authored_template_ngrams
    if included_template_ngram_overlap:
        raise ValueError("included authored templates overlap protected text n-grams")
    output_fingerprints = [full_text_fingerprint(record["tokens"]) for record in output]
    protected_overlap = set(output_fingerprints) & protected_fingerprints
    denylisted_overlap = set(output_fingerprints) & denylisted_fingerprints
    if protected_overlap or denylisted_overlap:
        raise ValueError("generated speech data overlaps a protected or denylisted full text")
    for record in output:
        validate_bio_record(record)
    labels = Counter(tag for record in output for tag in record["ner_tags"])
    entities = Counter(
        _entity_type(tag) for record in output for tag in record["ner_tags"] if tag.startswith("B-")
    )
    return output, {
        "allowed_transformations": [
            "lowercase",
            "o_punctuation_removed",
            "temporal_spoken",
            "filler",
        ],
        "augmented_record_count": len(output) - len(originals),
        "output_record_count": len(output),
        "source_record_count": len(source_records),
        "retained_source_record_count": len(originals),
        "excluded_protected_source_record_count": len(excluded_source_ids),
        "excluded_protected_source_ids_sha256": hashlib.sha256(
            "\n".join(excluded_source_ids).encode()
        ).hexdigest(),
        "protected_authored_template_ngram_count": len(protected_authored_template_ngrams),
        "protected_authored_template_ngrams_sha256": _ngram_digest(
            protected_authored_template_ngrams
        ),
        "excluded_protected_authored_template_count": excluded_template_count,
        "excluded_protected_authored_template_ids_sha256": hashlib.sha256(
            "\n".join(excluded_template_ids).encode()
        ).hexdigest(),
        "protected_authored_template_ngram_overlap_count": len(included_template_ngram_overlap),
        "speech_variant_count": len(source_ids),
        "template_counts": dict(sorted(template_counts.items())),
        "transformation_counts": dict(sorted(transform_counts.items())),
        "seed": seed,
        "selected_source_ids_sha256": hashlib.sha256("\n".join(source_ids).encode()).hexdigest(),
        "source_ids_sha256": hashlib.sha256(
            "\n".join(str(record["id"]) for record in originals).encode()
        ).hexdigest(),
        "label_counts": dict(sorted(labels.items())),
        "entity_counts": dict(sorted((key or "O", value) for key, value in entities.items())),
        "protected_overlap_count": 0,
        "denylisted_overlap_count": 0,
    }


def _read_jsonl(path: Path, *, protected: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{number} is not an object")
        if protected and not isinstance(value.get("text", value.get("tokens")), str | list):
            raise ValueError(f"protected record has no text at {path}:{number}")
        records.append(value)
    return records


def protected_fingerprints(paths: Sequence[Path]) -> tuple[frozenset[str], list[dict[str, str]]]:
    fingerprints: set[str] = set()
    provenance: list[dict[str, str]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"protected JSONL is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for record in _read_jsonl(path, protected=True):
            # Deliberately access only text/tokens, never held-out entity labels.
            fingerprints.add(full_text_fingerprint(record.get("text", record.get("tokens", []))))
        provenance.append({"path": str(path), "sha256": digest})
    return frozenset(fingerprints), provenance


def protected_authored_template_ngrams(paths: Sequence[Path]) -> frozenset[tuple[str, ...]]:
    """Build protected 3--5-grams from held-out text/tokens, never labels."""
    ngrams: set[tuple[str, ...]] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"protected JSONL is missing: {path}")
        for record in _read_jsonl(path, protected=True):
            # Deliberately access only text/tokens, never held-out entity labels.
            ngrams.update(normalized_ngrams(record.get("text", record.get("tokens", []))))
    return frozenset(ngrams)


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    temporary.replace(path)


def build_speech_training_file(
    input_path: Path,
    output_path: Path,
    statistics_path: Path,
    *,
    seed: int = DEFAULT_SEED,
    protected_paths: Sequence[Path] = DEFAULT_PROTECTED_PATHS,
    denylisted_fingerprints: frozenset[str] = KNOWN_OGG_TRANSCRIPT_FINGERPRINTS,
) -> dict[str, Any]:
    """Build auditable speech augmentation without writing to any protected path."""
    resolved = {path.resolve() for path in (*protected_paths, input_path)}
    if (
        output_path.resolve() in resolved
        or statistics_path.resolve() in resolved
        or output_path.resolve() == statistics_path.resolve()
    ):
        raise ValueError("speech output/statistics must not overwrite source or protected paths")
    fingerprints, protected_provenance = protected_fingerprints(
        tuple(Path(path) for path in protected_paths)
    )
    template_ngrams = protected_authored_template_ngrams(
        tuple(Path(path) for path in protected_paths)
    )
    source_records = _read_jsonl(input_path)
    protected_source_overlaps: list[dict[str, Any]] = []
    for protected_path, provenance in zip(protected_paths, protected_provenance, strict=True):
        path_fingerprints, _ = protected_fingerprints((Path(protected_path),))
        overlap_ids = [
            str(record["id"])
            for record in source_records
            if full_text_fingerprint(record["tokens"]) in path_fingerprints
        ]
        protected_source_overlaps.append(
            {
                **provenance,
                "source_record_count": len(overlap_ids),
                "source_ids_sha256": hashlib.sha256("\n".join(overlap_ids).encode()).hexdigest(),
            }
        )
    output, statistics = build_speech_records(
        source_records,
        seed=seed,
        protected_fingerprints=fingerprints,
        protected_authored_template_ngrams=template_ngrams,
        denylisted_fingerprints=denylisted_fingerprints,
    )
    _atomic_jsonl(output_path, output)
    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    result = {
        **statistics,
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "protected_paths": protected_provenance,
        "protected_source_overlaps": protected_source_overlaps,
        "protected_overlap_count": 0,
        "known_ogg_denylist_fingerprint_count": len(denylisted_fingerprints),
    }
    _atomic_json(statistics_path, result)
    return result
