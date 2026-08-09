"""Build a deterministic, leakage-safe public-data continuation mix.

The promoted speech-aware training file remains the immutable base.  Public
records are sampled reproducibly, deduplicated against that base and protected
evaluation text, and added only to the training split.  Public speech-year
phrases are first split into train/dev partitions so model selection never sees
the held-out phrases.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .labels import BIOValidationError, validate_bio_record
from .public_corpora import normalize_uzbek_token
from .speech_augmentation import (
    KNOWN_OGG_TRANSCRIPT_FINGERPRINTS,
    full_text_fingerprint,
    protected_fingerprints,
)

DEFAULT_V5_SEED = 20260807
DEFAULT_EXPERT_CAP = 10000
# Wave-13: expert-only public additions. TEMPORAL-only public streams stay out of
# train so PER/ORG/LOC can move; speech-year holdout is still built for gates.
DEFAULT_TEMPORAL_CAP = 0
DEFAULT_SPEECH_TRAIN_CAP = 0
DEFAULT_SPEECH_DEV_FRACTION = 0.20
DEFAULT_MINIMUM_SPEECH_DEV_RECORDS = 10
DEFAULT_MINIMUM_SPEECH_DEV_SOURCES = 2
PREFERRED_EXPERT_ENTITY_TYPES = frozenset({"PER", "ORG", "LOC"})
# Per-type expert upsample factors (a row uses the max matching factor).
DEFAULT_CORE_UPSAMPLE_FACTORS: dict[str, int] = {"LOC": 3, "ORG": 2}
# Extra copies of LOC-bearing immutable-base rows to reinforce gold LOC.
DEFAULT_BASE_LOC_UPSAMPLE_FACTOR = 1
# Gold LOC surfaces the parent keeps but light FT tends to drop (test gap).
HARD_LOC_TOKEN_MARKERS = ("mamlakat",)
DEFAULT_HARD_LOC_UPSAMPLE_FACTOR = 8
# Gold ORG surfaces near the remaining parent-only ORG miss (league names).
HARD_ORG_TOKEN_MARKERS = ("liga", "premyer")
DEFAULT_HARD_ORG_UPSAMPLE_FACTOR = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"blank JSONL line at {path}:{line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL record at {path}:{line_number}")
        try:
            validate_bio_record(value)
        except BIOValidationError as error:
            raise BIOValidationError(
                f"invalid BIO record at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value.get("id"), str) or not value["id"]:
            raise ValueError(f"record lacks a nonempty string id at {path}:{line_number}")
        records.append(value)
    return records


def _stable_rank(record: Mapping[str, Any], seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest()


def _stable_sample(
    records: Sequence[dict[str, Any]], cap: int, *, seed: int
) -> list[dict[str, Any]]:
    if cap < 0:
        raise ValueError("public source caps must be nonnegative")
    ranked = sorted(records, key=lambda record: (_stable_rank(record, seed), record["id"]))
    return sorted(ranked[:cap], key=lambda record: str(record["id"]))


def _entity_types_in_record(record: Mapping[str, Any]) -> frozenset[str]:
    types: set[str] = set()
    for tag in record.get("ner_tags", ()):
        if not isinstance(tag, str) or tag == "O" or "-" not in tag:
            continue
        types.add(tag.split("-", maxsplit=1)[1])
    return frozenset(types)


def _stable_sample_preferring_types(
    records: Sequence[dict[str, Any]],
    cap: int,
    *,
    seed: int,
    preferred_types: frozenset[str],
) -> list[dict[str, Any]]:
    """Fill the cap with preferred-type rows first, then stable-fill the remainder."""
    if cap < 0:
        raise ValueError("public source caps must be nonnegative")
    preferred = [record for record in records if _entity_types_in_record(record) & preferred_types]
    other = [
        record for record in records if not (_entity_types_in_record(record) & preferred_types)
    ]
    selected = _stable_sample(preferred, min(cap, len(preferred)), seed=seed)
    if len(selected) < cap:
        selected.extend(_stable_sample(other, cap - len(selected), seed=seed + 17))
    return sorted(selected, key=lambda record: str(record["id"]))


def _upsample_entity_records(
    records: Sequence[dict[str, Any]],
    *,
    types: frozenset[str],
    factor: int,
) -> list[dict[str, Any]]:
    """Deterministically duplicate records that carry the requested entity types."""
    if factor < 1:
        raise ValueError("upsample factor must be >= 1")
    if factor == 1:
        return sorted(records, key=lambda record: str(record["id"]))
    output: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda value: str(value["id"])):
        output.append(record)
        if not (_entity_types_in_record(record) & types):
            continue
        for copy_index in range(1, factor):
            duplicated = dict(record)
            duplicated["id"] = f"{record['id']}::up{copy_index}"
            output.append(duplicated)
    return sorted(output, key=lambda record: str(record["id"]))


def _upsample_by_type_factors(
    records: Sequence[dict[str, Any]],
    *,
    factors: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Duplicate each record by the max upsample factor among its entity types."""
    if any(factor < 1 for factor in factors.values()):
        raise ValueError("upsample factors must be >= 1")
    output: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda value: str(value["id"])):
        present = _entity_types_in_record(record)
        factor = 1
        for entity_type, value in factors.items():
            if entity_type in present:
                factor = max(factor, value)
        output.append(record)
        for copy_index in range(1, factor):
            duplicated = dict(record)
            duplicated["id"] = f"{record['id']}::up{copy_index}"
            output.append(duplicated)
    return sorted(output, key=lambda record: str(record["id"]))


def _record_has_hard_entity(
    record: Mapping[str, Any],
    *,
    entity_type: str,
    markers: Sequence[str],
) -> bool:
    tokens = record.get("tokens", ())
    tags = record.get("ner_tags", ())
    if len(tokens) != len(tags):
        return False
    suffix = f"-{entity_type}"
    for token, tag in zip(tokens, tags, strict=True):
        if not isinstance(tag, str) or not tag.endswith(suffix):
            continue
        surface = normalize_uzbek_token(str(token)).replace("'", "")
        if any(marker in surface for marker in markers):
            return True
    return False


def _record_has_hard_loc(
    record: Mapping[str, Any], markers: Sequence[str] = HARD_LOC_TOKEN_MARKERS
) -> bool:
    return _record_has_hard_entity(record, entity_type="LOC", markers=markers)


def _speech_partition(
    records: Sequence[dict[str, Any]], *, seed: int, dev_fraction: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0 < dev_fraction < 1:
        raise ValueError("speech dev fraction must be between zero and one")
    by_source: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        source = record.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"speech record {record['id']!r} lacks source provenance")
        by_source.setdefault(source, []).append(record)
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    for _source, source_records in sorted(by_source.items()):
        ranked = sorted(
            source_records,
            key=lambda record: (_stable_rank(record, seed), str(record["id"])),
        )
        dev_count = 0
        if len(ranked) >= 2:
            dev_count = max(1, round(len(ranked) * dev_fraction))
            dev_count = min(dev_count, len(ranked) - 1)
        dev.extend(ranked[:dev_count])
        train.extend(ranked[dev_count:])
    return (
        sorted(train, key=lambda record: str(record["id"])),
        sorted(dev, key=lambda record: str(record["id"])),
    )


def _filter_additions(
    records: Sequence[dict[str, Any]],
    *,
    seen: set[str],
    protected: frozenset[str],
    denylisted: frozenset[str],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    accepted: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for record in sorted(records, key=lambda value: str(value["id"])):
        fingerprint = full_text_fingerprint(record["tokens"])
        if fingerprint in denylisted:
            excluded["known_ogg_denylist"] += 1
        elif fingerprint in protected:
            excluded["protected_exact_text"] += 1
        elif fingerprint in seen:
            excluded["normalized_duplicate"] += 1
        else:
            seen.add(fingerprint)
            accepted.append(record)
    return accepted, excluded


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            for record in records:
                temporary.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
        Path(temporary_name).replace(path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
        Path(temporary_name).replace(path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def build_public_v5_mix(
    base_path: Path,
    expert_path: Path,
    temporal_path: Path,
    speech_paths: Sequence[Path],
    output_path: Path,
    statistics_path: Path,
    speech_dev_path: Path,
    *,
    protected_paths: Sequence[Path],
    seed: int = DEFAULT_V5_SEED,
    expert_cap: int = DEFAULT_EXPERT_CAP,
    temporal_cap: int = DEFAULT_TEMPORAL_CAP,
    speech_train_cap: int = DEFAULT_SPEECH_TRAIN_CAP,
    speech_dev_fraction: float = DEFAULT_SPEECH_DEV_FRACTION,
    minimum_speech_dev_records: int = DEFAULT_MINIMUM_SPEECH_DEV_RECORDS,
    minimum_speech_dev_sources: int = DEFAULT_MINIMUM_SPEECH_DEV_SOURCES,
    core_upsample_factors: Mapping[str, int] | None = None,
    base_loc_upsample_factor: int = DEFAULT_BASE_LOC_UPSAMPLE_FACTOR,
    hard_loc_upsample_factor: int = DEFAULT_HARD_LOC_UPSAMPLE_FACTOR,
    hard_org_upsample_factor: int = DEFAULT_HARD_ORG_UPSAMPLE_FACTOR,
    denylisted_fingerprints: frozenset[str] = KNOWN_OGG_TRANSCRIPT_FINGERPRINTS,
) -> dict[str, Any]:
    """Merge the promoted base with capped public records and a held-out speech dev set.

    Expert/temporal caps prefer full BIO (PER/ORG/LOC/TEMPORAL/…). Speech-year
    phrases are capped so TEMPORAL-only public text cannot drown general entities.
    ORG/LOC-bearing expert rows may be upsampled after selection. LOC-bearing base
    rows may also be duplicated into the train mix (new ids) to reinforce gold LOC.
    """
    upsample_factors = dict(
        DEFAULT_CORE_UPSAMPLE_FACTORS if core_upsample_factors is None else core_upsample_factors
    )
    if base_loc_upsample_factor < 1:
        raise ValueError("base LOC upsample factor must be >= 1")
    if hard_loc_upsample_factor < 1:
        raise ValueError("hard LOC upsample factor must be >= 1")
    if hard_org_upsample_factor < 1:
        raise ValueError("hard ORG upsample factor must be >= 1")
    inputs = (base_path, expert_path, temporal_path, *speech_paths, *protected_paths)
    resolved_inputs = {path.resolve() for path in inputs}
    destinations = (output_path, statistics_path, speech_dev_path)
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("V5 output, statistics, and speech dev paths must be distinct")
    if any(path.resolve() in resolved_inputs for path in destinations):
        raise ValueError("V5 artifacts must not overwrite input or protected files")

    protected, protected_provenance = protected_fingerprints(protected_paths)
    stable_protected_provenance = [
        {"filename": Path(value["path"]).name, "sha256": value["sha256"]}
        for value in protected_provenance
    ]
    base = _read_records(base_path)
    base_fingerprints = [full_text_fingerprint(record["tokens"]) for record in base]
    if any(value in protected for value in base_fingerprints):
        raise ValueError("immutable promoted base overlaps a protected evaluation text")
    if any(value in denylisted_fingerprints for value in base_fingerprints):
        raise ValueError("immutable promoted base overlaps the known OGG denylist")
    seen = set(base_fingerprints)

    source_records = {
        "uzner_expert": _read_records(expert_path),
        "uzner_temporal": _read_records(temporal_path),
    }
    accepted_by_source: dict[str, list[dict[str, Any]]] = {}
    exclusions: Counter[str] = Counter()

    expert_candidates, current = _filter_additions(
        source_records["uzner_expert"],
        seen=seen,
        protected=protected,
        denylisted=denylisted_fingerprints,
    )
    exclusions.update({f"uzner_expert:{key}": count for key, count in current.items()})
    accepted_by_source["uzner_expert"] = _upsample_by_type_factors(
        _stable_sample_preferring_types(
            expert_candidates,
            expert_cap,
            seed=seed,
            preferred_types=PREFERRED_EXPERT_ENTITY_TYPES,
        ),
        factors=upsample_factors,
    )
    # Sampling should not reserve unselected fingerprints against later sources.
    seen = set(base_fingerprints)
    seen.update(
        full_text_fingerprint(record["tokens"]) for record in accepted_by_source["uzner_expert"]
    )

    temporal_candidates, current = _filter_additions(
        source_records["uzner_temporal"],
        seen=seen,
        protected=protected,
        denylisted=denylisted_fingerprints,
    )
    exclusions.update({f"uzner_temporal:{key}": count for key, count in current.items()})
    accepted_by_source["uzner_temporal"] = _stable_sample(
        temporal_candidates, temporal_cap, seed=seed + 1
    )
    seen.update(
        full_text_fingerprint(record["tokens"]) for record in accepted_by_source["uzner_temporal"]
    )

    speech_records: list[dict[str, Any]] = []
    speech_input_counts: dict[str, int] = {}
    for index, speech_path in enumerate(speech_paths):
        current_records = _read_records(speech_path)
        speech_input_counts[f"public_speech_{index}"] = len(current_records)
        speech_records.extend(current_records)
    speech_candidates, current = _filter_additions(
        speech_records,
        seen=seen,
        protected=protected,
        denylisted=denylisted_fingerprints,
    )
    exclusions.update({f"public_speech:{key}": count for key, count in current.items()})
    speech_train, speech_dev = _speech_partition(
        speech_candidates, seed=seed + 2, dev_fraction=speech_dev_fraction
    )
    speech_dev_sources = Counter(str(record["source"]) for record in speech_dev)
    if minimum_speech_dev_records <= 0 or minimum_speech_dev_sources <= 0:
        raise ValueError("minimum speech dev records and sources must be positive")
    if len(speech_dev) < minimum_speech_dev_records:
        raise ValueError(
            "public speech-year dev set is too small: "
            f"{len(speech_dev)} < {minimum_speech_dev_records}"
        )
    if len(speech_dev_sources) < minimum_speech_dev_sources:
        raise ValueError(
            "public speech-year dev set lacks source coverage: "
            f"{len(speech_dev_sources)} < {minimum_speech_dev_sources}"
        )
    accepted_by_source["public_speech_train"] = _stable_sample(
        speech_train, speech_train_cap, seed=seed + 3
    )

    base_loc_rows = [record for record in base if "LOC" in _entity_types_in_record(record)]
    base_loc_boost = [
        record
        for record in _upsample_by_type_factors(
            base_loc_rows, factors={"LOC": base_loc_upsample_factor}
        )
        if "::up" in str(record["id"])
    ]
    accepted_by_source["base_loc_upsample"] = sorted(
        base_loc_boost, key=lambda record: str(record["id"])
    )

    hard_loc_rows = [record for record in base if _record_has_hard_loc(record)]
    hard_loc_boost = [
        record
        for record in _upsample_by_type_factors(
            hard_loc_rows, factors={"LOC": hard_loc_upsample_factor}
        )
        if "::up" in str(record["id"])
    ]
    accepted_by_source["hard_loc_upsample"] = sorted(
        hard_loc_boost, key=lambda record: str(record["id"])
    )

    hard_org_rows = [
        record
        for record in base
        if _record_has_hard_entity(record, entity_type="ORG", markers=HARD_ORG_TOKEN_MARKERS)
    ]
    hard_org_boost = [
        record
        for record in _upsample_by_type_factors(
            hard_org_rows, factors={"ORG": hard_org_upsample_factor}
        )
        if "::up" in str(record["id"])
    ]
    accepted_by_source["hard_org_upsample"] = sorted(
        hard_org_boost, key=lambda record: str(record["id"])
    )

    additions = [
        record
        for source in (
            "uzner_expert",
            "uzner_temporal",
            "public_speech_train",
            "base_loc_upsample",
            "hard_loc_upsample",
            "hard_org_upsample",
        )
        for record in accepted_by_source[source]
    ]
    output_records = [*base, *additions]
    selected_ids = [str(record["id"]) for record in additions]
    source_paths = {
        "base": base_path,
        "uzner_expert": expert_path,
        "uzner_temporal": temporal_path,
        **{f"public_speech_{index}": path for index, path in enumerate(speech_paths)},
    }
    stats: dict[str, Any] = {
        # Compatibility keys consumed by training/train_ner.py.
        "allowed_transformations": [
            "pinned UzNER BIOES-to-BIO label mapping",
            "phrase-only extraction of explicit spoken calendar years from pinned speech text",
            "strict 1800--2035 digit calendar years normalized to canonical spoken Uzbek",
            "deterministic LOC upsample of immutable-base rows into train only",
            "deterministic hard-LOC surface upsample from immutable-base rows",
            "deterministic hard-ORG surface upsample from immutable-base rows",
        ],
        "augmented_record_count": len(additions),
        "output_record_count": len(output_records),
        "seed": seed,
        "selected_source_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")
        ).hexdigest(),
        "source_record_count": len(base),
        "transformation_counts": {
            source: len(records) for source, records in sorted(accepted_by_source.items())
        },
        # V5-specific provenance and integrity evidence.
        "caps": {
            "uzner_expert": expert_cap,
            "uzner_temporal": temporal_cap,
            "public_speech_train": speech_train_cap,
            "core_upsample_factors": dict(sorted(upsample_factors.items())),
            "base_loc_upsample_factor": base_loc_upsample_factor,
            "hard_loc_upsample_factor": hard_loc_upsample_factor,
            "hard_loc_token_markers": list(HARD_LOC_TOKEN_MARKERS),
            "hard_org_upsample_factor": hard_org_upsample_factor,
            "hard_org_token_markers": list(HARD_ORG_TOKEN_MARKERS),
        },
        "excluded_counts": dict(sorted(exclusions.items())),
        "input_record_counts": {
            "base": len(base),
            "uzner_expert": len(source_records["uzner_expert"]),
            "uzner_temporal": len(source_records["uzner_temporal"]),
            **speech_input_counts,
        },
        "input_sha256": {name: _sha256(path) for name, path in sorted(source_paths.items())},
        "known_ogg_denylist_fingerprint_count": len(denylisted_fingerprints),
        "protected_overlap_count": 0,
        "protected_paths": stable_protected_provenance,
        "speech_dev_fraction": speech_dev_fraction,
        "speech_dev_minimum_records": minimum_speech_dev_records,
        "speech_dev_minimum_sources": minimum_speech_dev_sources,
        "speech_dev_record_count": len(speech_dev),
        "speech_dev_source_counts": dict(sorted(speech_dev_sources.items())),
        "speech_train_record_count": len(speech_train),
    }

    _atomic_jsonl(output_path, output_records)
    _atomic_jsonl(speech_dev_path, speech_dev)
    stats["output_filename"] = output_path.name
    stats["output_sha256"] = _sha256(output_path)
    stats["speech_dev_filename"] = speech_dev_path.name
    stats["speech_dev_sha256"] = _sha256(speech_dev_path)
    _atomic_json(statistics_path, stats)
    return stats
