"""Prepare public Uzbek corpora and continue NER fine-tuning in parallel on Modal."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import modal

from uzbek_speech_entities.ner.public_corpora import (
    COMMON_VOICE_UZ,
    FLEURS_UZ,
    GLOTCC_UZN_LATN,
    IT_YOUTUBE_UZ,
    LEGAL_UZ,
    NEWS_YOUTUBE_UZ,
    OASST2_UZ,
    PODCASTS_TASHKENT_UZ,
    USC_UZ,
    UZBEK_NEWS_TEXT,
    UZBEKVOICE2_UZ,
    UZBEKVOICE_UZ,
    UZNER_SHA256,
    UZNER_URL,
    WIKIPEDIA_UZ,
    ZERO_SHOT_NEWS_UZ,
    SpeechCorpus,
    discover_parquet_urls,
    import_uzner_workbook,
    iter_parquet_text_column,
    merge_spoken_year_record_batches,
    mine_spoken_year_records,
    verify_hf_revision,
)
from uzbek_speech_entities.ner.public_mix import build_public_v5_mix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace")
REMOTE_CONFIG_DIR = REMOTE_ROOT / "configs"
REMOTE_DATA_DIR = REMOTE_ROOT / "data" / "processed" / "ner"
REMOTE_FIXTURE = REMOTE_ROOT / "tests" / "fixtures" / "speech_ner_eval.jsonl"
REMOTE_CHECKPOINT = REMOTE_ROOT / "models" / "ner" / "final"
REMOTE_PUBLIC_ROOT = Path("/public-data")
REMOTE_OUTPUT_ROOT = Path("/outputs")

DATA_VOLUME_NAME = "uzbek-speech-ner-public-data-v5"
OUTPUT_VOLUME_NAME = "uzbek-speech-ner-public-v5-runs"
RESERVATION_DICT_NAME = "uzbek-speech-ner-public-v5-reservations"
DEFAULT_RELEASE = "public-ner-v5n24-20260808"
# Wave-24 parent: best v5n5 PASS; hard-LOC + hard-ORG (liga) upsample.
LOCAL_PARENT_DIR = (
    PROJECT_ROOT
    / "models"
    / "ner"
    / "candidates"
    / "public-v5n5-ft5-lr3e6-ep1-seed2"
)
EXPECTED_PARENT_MODEL_SHA256 = (
    "498d6bf27179d49af48cfb264bd6eb5cf534f9ed507eaf4435e275157888b976"
)
PARENT_CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)
INFERENCE_BUNDLE_FILES = PARENT_CHECKPOINT_FILES | {"labels.json"}
TEMPLATE_CONFIG_BASENAME = "ner_public_v5_seed1.yaml"


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int
    learning_rate: float
    epochs: int
    output_name: str
    weight_decay: float = 0.01


def _approved_experiments() -> dict[str, ExperimentSpec]:
    """Return wave-24: hard-LOC+ORG boost; seed-pairs at lr4e7-ep2."""
    experiments: dict[str, ExperimentSpec] = {}
    seed_pairs = (
        (20261063, 20261064),
        (20261065, 20261066),
        (20261067, 20261068),
        (20261069, 20261070),
        (20261071, 20261072),
        (20261073, 20261074),
        (20261075, 20261076),
        (20261077, 20261078),
    )
    for pair_index, (seed1, seed2) in enumerate(seed_pairs, start=1):
        pair_label = f"p{pair_index}"
        for seed_label, seed in (("seed1", seed1), ("seed2", seed2)):
            basename = f"ner_public_v5_ft24_lr4e7_ep2_{pair_label}_{seed_label}.yaml"
            output_name = f"public-v5n24-ft24-lr4e7-ep2-{pair_label}-{seed_label}"
            experiments[basename] = ExperimentSpec(
                seed=seed,
                learning_rate=0.0000004,
                epochs=2,
                output_name=output_name,
                weight_decay=0.01,
            )
    return experiments


APPROVED_CONFIGS = _approved_experiments()
JOB_NAMES = (
    "uzner-expert",
    "uzner-temporal",
    "common-voice",
    "usc",
    "news-youtube",
    "it-youtube",
    "podcasts-tashkent",
    "uzbekvoice",
    "uzbekvoice2",
    "uzbek-news-text",
    "wikipedia-uz",
    "fleurs-uz",
    "glotcc-uzn-latn",
    "zero-shot-news",
    "oasst2-uz",
)
_SAFE_NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,80}\Z")


data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fsspec>=2024.0,<2027",
        "numpy>=1.26",
        "openpyxl>=3.1,<4",
        "pyarrow>=15,<24",
        "pydantic>=2,<3",
    )
    .env(
        {
            "PYTHONPATH": ":".join(
                (str(REMOTE_ROOT / "training"), str(REMOTE_ROOT / "src"))
            )
        }
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
    .add_local_dir(PROJECT_ROOT / "training", remote_path=str(REMOTE_ROOT / "training"))
    .add_local_file(
        PROJECT_ROOT / "data" / "processed" / "ner" / "train_speech_augmented.jsonl",
        remote_path=str(REMOTE_DATA_DIR / "train_speech_augmented.jsonl"),
    )
    .add_local_file(
        PROJECT_ROOT / "data" / "processed" / "ner" / "validation.jsonl",
        remote_path=str(REMOTE_DATA_DIR / "validation.jsonl"),
    )
    .add_local_file(
        PROJECT_ROOT / "data" / "processed" / "ner" / "test.jsonl",
        remote_path=str(REMOTE_DATA_DIR / "test.jsonl"),
    )
    .add_local_file(
        PROJECT_ROOT / "tests" / "fixtures" / "speech_ner_eval.jsonl",
        remote_path=str(REMOTE_FIXTURE),
    )
)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-modal.txt"))
    .env(
        {
            "PYTHONPATH": ":".join(
                (str(REMOTE_ROOT), str(REMOTE_ROOT / "training"), str(REMOTE_ROOT / "src"))
            )
        }
    )
    .add_local_dir(PROJECT_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
    .add_local_dir(PROJECT_ROOT / "training", remote_path=str(REMOTE_ROOT / "training"))
    .add_local_dir(PROJECT_ROOT / "configs", remote_path=str(REMOTE_CONFIG_DIR))
    .add_local_dir(
        PROJECT_ROOT / "data" / "processed" / "ner",
        remote_path=str(REMOTE_DATA_DIR),
    )
)
for checkpoint_file in sorted(PARENT_CHECKPOINT_FILES):
    training_image = training_image.add_local_file(
        LOCAL_PARENT_DIR / checkpoint_file,
        remote_path=str(REMOTE_CHECKPOINT / checkpoint_file),
    )
training_image = training_image.add_local_file(
    LOCAL_PARENT_DIR / "labels.json",
    remote_path=str(REMOTE_CHECKPOINT / "labels.json"),
).add_local_file(
    PROJECT_ROOT / "tests" / "fixtures" / "speech_ner_eval.jsonl",
    remote_path=str(REMOTE_FIXTURE),
)

app = modal.App("uzbek-speech-ner-public-v5")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True, version=2)
reservation_dict = modal.Dict.from_name(RESERVATION_DICT_NAME, create_if_missing=True)


def validate_release_name(release: str) -> str:
    if not _SAFE_NAME.fullmatch(release):
        raise ValueError("release must contain only lowercase letters, digits, and hyphens")
    return release


def selected_config_basenames(seed: str) -> tuple[str, ...]:
    selected = {
        "seed1": ("ner_public_v5_ft24_lr4e7_ep2_p1_seed1.yaml",),
        "seed2": ("ner_public_v5_ft24_lr4e7_ep2_p1_seed2.yaml",),
        "both": tuple(APPROVED_CONFIGS),
        "sweep": tuple(APPROVED_CONFIGS),
    }
    try:
        return selected[seed]
    except KeyError as error:
        raise ValueError("seed must be one of: seed1, seed2, both, sweep") from error


def speech_corpus(source: str) -> SpeechCorpus:
    selected = {
        "common-voice": COMMON_VOICE_UZ,
        "usc": USC_UZ,
        "news-youtube": NEWS_YOUTUBE_UZ,
        "it-youtube": IT_YOUTUBE_UZ,
        "podcasts-tashkent": PODCASTS_TASHKENT_UZ,
        "uzbekvoice": UZBEKVOICE_UZ,
        "uzbekvoice2": UZBEKVOICE2_UZ,
        "uzbek-news-text": UZBEK_NEWS_TEXT,
        "wikipedia-uz": WIKIPEDIA_UZ,
        "fleurs-uz": FLEURS_UZ,
        "glotcc-uzn-latn": GLOTCC_UZN_LATN,
        "legal-uz": LEGAL_UZ,
        "zero-shot-news": ZERO_SHOT_NEWS_UZ,
        "oasst2-uz": OASST2_UZ,
    }
    try:
        return selected[source]
    except KeyError as error:
        raise ValueError(
            "speech source must be common-voice, usc, news-youtube, it-youtube, "
            "podcasts-tashkent, uzbekvoice, uzbekvoice2, uzbek-news-text, wikipedia-uz, "
            "fleurs-uz, glotcc-uzn-latn, legal-uz, zero-shot-news, or oasst2-uz"
        ) from error


def rewritten_remote_config(
    config_basename: str,
    values: Mapping[str, Any],
    release: str = DEFAULT_RELEASE,
) -> dict[str, Any]:
    if Path(config_basename).name != config_basename or config_basename not in APPROVED_CONFIGS:
        raise ValueError("only an approved public V5 experiment may run on Modal")
    rewritten = copy.deepcopy(dict(values))
    model = rewritten.get("model")
    output = rewritten.get("output")
    if not isinstance(model, dict) or not isinstance(output, dict):
        raise ValueError("approved config is missing model or output sections")
    spec = APPROVED_CONFIGS[config_basename]
    training = rewritten.get("training")
    if not isinstance(training, dict):
        raise ValueError("approved config is missing training section")
    training["seed"] = spec.seed
    training["learning_rate"] = spec.learning_rate
    training["epochs"] = spec.epochs
    training["weight_decay"] = spec.weight_decay
    model["checkpoint"] = str(REMOTE_CHECKPOINT)
    output["directory"] = str(
        REMOTE_OUTPUT_ROOT / validate_release_name(release) / spec.output_name
    )
    return rewritten


def collect_call_results(calls: Sequence[tuple[str, Any]]) -> list[Any]:
    """Wait for every already-spawned Modal call and report all failures together."""
    results: list[Any] = []
    errors: list[str] = []
    for name, call in calls:
        try:
            results.append(call.get())
        except Exception as error:  # noqa: BLE001 - aggregate remote failures with job names.
            errors.append(f"{name}: {type(error).__name__}: {error}")
    if errors:
        raise RuntimeError("parallel Modal call(s) failed: " + " | ".join(errors))
    return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _manifest_for(directory: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    files = {
        path.relative_to(directory).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    return {"complete": True, "files": files, **metadata}


def _verify_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"completion manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError(f"invalid completion manifest: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"manifest has no files: {manifest_path}")
    expected_paths = set(files)
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_paths != expected_paths:
        raise ValueError(
            f"manifest file set mismatch at {directory}: {actual_paths ^ expected_paths}"
        )
    for relative, expected in files.items():
        if not isinstance(relative, str) or not isinstance(expected, dict):
            raise ValueError(f"invalid manifest entry at {manifest_path}")
        path = directory / relative
        if path.stat().st_size != expected.get("bytes") or _sha256(path) != expected.get(
            "sha256"
        ):
            raise ValueError(f"manifest integrity mismatch: {path}")
    return manifest


def _publish_directory(
    local_directory: Path, remote_directory: Path, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    if remote_directory.exists():
        return _verify_manifest(remote_directory)
    remote_directory.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_directory, remote_directory)
    manifest = _manifest_for(remote_directory, metadata)
    # Publication sentinel is written only after every listed file exists.
    _write_json(remote_directory / "manifest.json", manifest)
    return manifest


def _download_uzner(destination: Path) -> None:
    request = Request(UZNER_URL, headers={"User-Agent": "uzbek-speech-ner-v5/1"})
    with urlopen(request, timeout=60) as response, destination.open("wb") as output:  # noqa: S310
        shutil.copyfileobj(response, output, length=1024 * 1024)
    actual = _sha256(destination)
    if actual != UZNER_SHA256:
        raise ValueError(f"downloaded UzNER checksum mismatch: {actual}")


def _write_remote_config(
    config_basename: str, release: str
) -> tuple[Path, dict[str, Any]]:
    import yaml

    if Path(config_basename).name != config_basename or config_basename not in APPROVED_CONFIGS:
        raise ValueError("only an approved public V5 experiment may run on Modal")
    source_path = REMOTE_CONFIG_DIR / config_basename
    if not source_path.is_file():
        source_path = REMOTE_CONFIG_DIR / TEMPLATE_CONFIG_BASENAME
    values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError(f"approved config is not a mapping: {config_basename}")
    rewritten = rewritten_remote_config(config_basename, values, release)
    path = Path("/tmp") / f"modal-{config_basename}"
    path.write_text(yaml.safe_dump(rewritten, sort_keys=False), encoding="utf-8")
    return path, rewritten


@app.function(
    image=data_image,
    timeout=60 * 60,
    memory=4096,
    volumes={str(REMOTE_PUBLIC_ROOT): data_volume},
    include_source=False,
)
def prepare_uzner(mode: str, release: str = DEFAULT_RELEASE) -> dict[str, Any]:
    """Download and import one independent UzNER slice."""
    validate_release_name(release)
    if mode not in {"expert", "temporal"}:
        raise ValueError("UzNER mode must be expert or temporal")
    job_directory = REMOTE_PUBLIC_ROOT / "jobs" / release / f"uzner-{mode}"
    data_volume.reload()
    if job_directory.exists():
        return _verify_manifest(job_directory)
    with tempfile.TemporaryDirectory(prefix=f"uzner-{mode}-") as temporary:
        local_directory = Path(temporary)
        workbook = local_directory / "UzNER-Style-v2.xlsx"
        _download_uzner(workbook)
        import_uzner_workbook(
            workbook,
            local_directory / "records.jsonl",
            local_directory / "statistics.json",
            mode=mode,
        )
        workbook.unlink()
        manifest = _publish_directory(
            local_directory,
            job_directory,
            {"job": f"uzner-{mode}", "release": release},
        )
    data_volume.commit()
    return manifest


@app.function(
    image=data_image,
    timeout=20 * 60,
    memory=2048,
    max_containers=64,
    include_source=False,
)
def mine_speech_shard(source: str, url: str) -> list[dict[str, object]]:
    """Mine one allowlisted transcript shard so sources scale out across containers."""
    corpus = speech_corpus(source)
    expected_prefix = f"https://huggingface.co/datasets/{corpus.dataset_id}/"
    if not url.startswith(expected_prefix):
        raise ValueError(f"unexpected Parquet URL for {corpus.dataset_id}: {url!r}")
    records, _statistics = mine_spoken_year_records(
        corpus, iter_parquet_text_column((url,), corpus.text_column)
    )
    return records


@app.function(
    image=data_image,
    timeout=60 * 60,
    memory=4096,
    volumes={str(REMOTE_PUBLIC_ROOT): data_volume},
    include_source=False,
)
def prepare_speech(source: str, release: str = DEFAULT_RELEASE) -> dict[str, Any]:
    """Mine one source in shard-parallel containers and publish it atomically."""
    validate_release_name(release)
    corpus = speech_corpus(source)
    job_directory = REMOTE_PUBLIC_ROOT / "jobs" / release / source
    data_volume.reload()
    if job_directory.exists():
        return _verify_manifest(job_directory)
    verify_hf_revision(corpus)
    urls = discover_parquet_urls(corpus)
    shard_calls = [
        (f"{source}-shard-{index:04d}", mine_speech_shard.spawn(source, url))
        for index, url in enumerate(urls)
    ]
    shard_records = collect_call_results(shard_calls)
    records, statistics = merge_spoken_year_record_batches(corpus, shard_records)
    statistics["parquet_urls"] = urls
    with tempfile.TemporaryDirectory(prefix=f"speech-{source}-") as temporary:
        local_directory = Path(temporary)
        _write_jsonl(local_directory / "records.jsonl", records)
        _write_json(local_directory / "statistics.json", statistics)
        manifest = _publish_directory(
            local_directory,
            job_directory,
            {"job": source, "release": release},
        )
    data_volume.commit()
    return manifest


@app.function(
    image=data_image,
    timeout=30 * 60,
    memory=4096,
    volumes={str(REMOTE_PUBLIC_ROOT): data_volume},
    include_source=False,
)
def finalize_release(
    release: str = DEFAULT_RELEASE, jobs_release: str = ""
) -> dict[str, Any]:
    """Merge only complete job outputs and publish one immutable V5 release.

    ``jobs_release`` may point at an earlier completed prep release so a new mix
    identity can reuse mined corpora without re-downloading Hugging Face shards.
    Per-job, prefer ``release`` artifacts when present, else fall back to
    ``jobs_release`` (lets UzNER be re-imported while speech jobs stay cached).
    """
    validate_release_name(release)
    jobs_name = validate_release_name(jobs_release) if jobs_release else release
    data_volume.reload()
    primary_root = REMOTE_PUBLIC_ROOT / "jobs" / release
    fallback_root = REMOTE_PUBLIC_ROOT / "jobs" / jobs_name

    def job_dir(job: str) -> Path:
        primary = primary_root / job
        if (primary / "manifest.json").is_file():
            return primary
        return fallback_root / job

    job_manifests = {job: _verify_manifest(job_dir(job)) for job in JOB_NAMES}
    release_directory = REMOTE_PUBLIC_ROOT / "releases" / release
    if release_directory.exists():
        return _verify_manifest(release_directory)
    with tempfile.TemporaryDirectory(prefix="public-v5-release-") as temporary:
        local_directory = Path(temporary)
        statistics = build_public_v5_mix(
            REMOTE_DATA_DIR / "train_speech_augmented.jsonl",
            job_dir("uzner-expert") / "records.jsonl",
            job_dir("uzner-temporal") / "records.jsonl",
            tuple(
                job_dir(name) / "records.jsonl"
                for name in JOB_NAMES
                if name not in {"uzner-expert", "uzner-temporal"}
            ),
            local_directory / "train_public_v5.jsonl",
            local_directory / "public_v5_statistics.json",
            local_directory / "speech_year_dev.jsonl",
            protected_paths=(
                REMOTE_DATA_DIR / "validation.jsonl",
                REMOTE_DATA_DIR / "test.jsonl",
                REMOTE_FIXTURE,
            ),
        )
        manifest = _publish_directory(
            local_directory,
            release_directory,
            {
                "job_manifest_sha256": {
                    job: hashlib.sha256(
                        json.dumps(value, sort_keys=True).encode()
                    ).hexdigest()
                    for job, value in sorted(job_manifests.items())
                },
                "jobs_release": jobs_name,
                "output_record_count": statistics["output_record_count"],
                "release": release,
                "speech_dev_record_count": statistics["speech_dev_record_count"],
                "caps": statistics.get("caps"),
            },
        )
    data_volume.commit()
    return manifest


def _validate_parent_checkpoint() -> None:
    actual = _sha256(REMOTE_CHECKPOINT / "model.safetensors")
    if actual != EXPECTED_PARENT_MODEL_SHA256:
        raise RuntimeError(
            "uploaded parent checkpoint does not match the expected wave parent: "
            f"{actual}"
        )


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def _verify_release_binding(
    manifest: Mapping[str, Any], release: str, release_manifest: Mapping[str, Any]
) -> None:
    if manifest.get("data_release") != release:
        raise ValueError(
            "cached seed belongs to a different data release: "
            f"{manifest.get('data_release')!r} != {release!r}"
        )
    expected_digest = _manifest_digest(release_manifest)
    if manifest.get("data_release_manifest_sha256") != expected_digest:
        raise ValueError("cached seed data-release manifest digest does not match")


def _install_release(release: str) -> dict[str, Any]:
    data_volume.reload()
    release_directory = REMOTE_PUBLIC_ROOT / "releases" / validate_release_name(release)
    manifest = _verify_manifest(release_directory)
    files = manifest["files"]
    for filename in ("train_public_v5.jsonl", "public_v5_statistics.json"):
        if filename not in files:
            raise ValueError(f"release is missing required training file: {filename}")
        destination = REMOTE_DATA_DIR / filename
        shutil.copyfile(release_directory / filename, destination)
        if _sha256(destination) != files[filename]["sha256"]:
            raise ValueError(f"copied release file failed verification: {filename}")
    return manifest


def _stage_inference_bundle(
    output_directory: Path,
    config_basename: str,
    release: str,
    release_manifest: Mapping[str, Any],
    speech_dev_path: Path,
) -> dict[str, Any]:
    pointer = json.loads((output_directory / "best_checkpoint.json").read_text(encoding="utf-8"))
    relative_checkpoint = pointer.get("checkpoint")
    if not isinstance(relative_checkpoint, str) or not relative_checkpoint:
        raise RuntimeError("training did not write a valid best-checkpoint pointer")
    checkpoint = output_directory / relative_checkpoint
    inference_directory = output_directory / "inference"
    if inference_directory.exists():
        raise FileExistsError(f"refusing to replace inference bundle: {inference_directory}")
    inference_directory.mkdir()
    for filename in sorted(INFERENCE_BUNDLE_FILES):
        source = output_directory / filename if filename == "labels.json" else checkpoint / filename
        if not source.is_file():
            raise FileNotFoundError(f"best checkpoint is missing inference file: {source}")
        shutil.copyfile(source, inference_directory / filename)
    _evaluate_checkpoint_reports(
        inference_directory,
        config_basename,
        speech_dev_path,
        inference_directory,
    )
    manifest = _manifest_for(
        inference_directory,
        {
            "data_release": release,
            "data_release_manifest_sha256": _manifest_digest(release_manifest),
            "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
            "source_checkpoint": relative_checkpoint,
        },
    )
    _write_json(inference_directory / "manifest.json", manifest)
    return manifest


def _evaluate_checkpoint_reports(
    checkpoint: Path,
    config_basename: str,
    speech_dev_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate clean, immutable speech, and held-out speech-year gates."""
    from training.evaluate_ner import evaluate as evaluate_clean
    from training.evaluate_public_speech_years import evaluate_public_speech_checkpoint
    from training.evaluate_speech_ner import evaluate_speech_checkpoint

    evaluation_config = REMOTE_CONFIG_DIR / config_basename
    if not evaluation_config.is_file():
        evaluation_config = REMOTE_CONFIG_DIR / TEMPLATE_CONFIG_BASENAME
    clean = evaluate_clean(
        checkpoint,
        checkpoint,
        evaluation_config,
        "test",
    )
    immutable_speech = evaluate_speech_checkpoint(
        checkpoint,
        REMOTE_FIXTURE,
        confidence_threshold=0.80,
    )
    public_speech_year = evaluate_public_speech_checkpoint(
        checkpoint,
        speech_dev_path,
        confidence_threshold=0.80,
    )
    reports = {
        "clean_test": clean,
        "immutable_speech_fixture": immutable_speech,
        "public_speech_year_dev": public_speech_year,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "clean_test_metrics.json", clean)
    _write_json(output_directory / "speech_fixture_metrics.json", immutable_speech)
    _write_json(output_directory / "speech_year_dev_metrics.json", public_speech_year)
    return reports


def _training_summary(
    config_basename: str, output_directory: Path, release: str
) -> dict[str, str | int | None]:
    spec = APPROVED_CONFIGS[config_basename]
    pointer = json.loads((output_directory / "best_checkpoint.json").read_text(encoding="utf-8"))
    return {
        "best_checkpoint": pointer.get("checkpoint"),
        "config": config_basename,
        "data_release": release,
        "inference_directory": str(output_directory / "inference"),
        "output_directory": str(output_directory),
        "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
        "epochs": spec.epochs,
        "learning_rate": str(spec.learning_rate),
        "seed": spec.seed,
    }


@app.function(
    image=training_image,
    gpu="L4",
    max_containers=32,
    timeout=2 * 60 * 60,
    volumes={
        str(REMOTE_PUBLIC_ROOT): data_volume,
        str(REMOTE_OUTPUT_ROOT): output_volume,
    },
    include_source=False,
)
def train_v5_seed(config_basename: str, release: str = DEFAULT_RELEASE) -> dict[str, Any]:
    """Continue one V5 seed from the configured wave parent and frozen data release."""
    _validate_parent_checkpoint()
    release_manifest = _install_release(release)
    config_path, rewritten = _write_remote_config(config_basename, release)
    output = rewritten["output"]
    assert isinstance(output, dict)
    output_directory = Path(str(output["directory"]))
    output_volume.reload()
    if (output_directory / "inference" / "manifest.json").is_file():
        cached_manifest = _verify_manifest(output_directory / "inference")
        _verify_release_binding(cached_manifest, release, release_manifest)
        return _training_summary(config_basename, output_directory, release)
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError(f"refusing partial or duplicate V5 run: {output_directory}")
    reservation_key = f"{release}:{config_basename}"
    if not reservation_dict.put(
        reservation_key,
        {"config": config_basename, "release": release, "status": "running"},
        skip_if_exists=True,
    ):
        raise RuntimeError(f"V5 seed is already reserved: {reservation_key}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                str(REMOTE_ROOT / "training" / "train_ner.py"),
                "--config",
                str(config_path),
            ],
            check=True,
            cwd=REMOTE_ROOT,
        )
        _stage_inference_bundle(
            output_directory,
            config_basename,
            release,
            release_manifest,
            REMOTE_PUBLIC_ROOT / "releases" / release / "speech_year_dev.jsonl",
        )
        subprocess.run(["sync", str(REMOTE_OUTPUT_ROOT)], check=True)
        output_volume.commit()
    except Exception as error:
        # Keep the reservation fail-closed so a retry cannot collide with a
        # partially committed checkpoint tree. Recovery requires inspection.
        reservation_dict.put(
            reservation_key,
            {
                "config": config_basename,
                "error_type": type(error).__name__,
                "release": release,
                "status": "failed",
            },
        )
        with suppress(Exception):
            subprocess.run(["sync", str(REMOTE_OUTPUT_ROOT)], check=True)
            output_volume.commit()
        raise
    reservation_dict.put(
        reservation_key,
        {"config": config_basename, "release": release, "status": "complete"},
    )
    return _training_summary(config_basename, output_directory, release)


@app.function(
    image=training_image,
    timeout=60 * 60,
    memory=4096,
    volumes={
        str(REMOTE_PUBLIC_ROOT): data_volume,
        str(REMOTE_OUTPUT_ROOT): output_volume,
    },
    include_source=False,
)
def evaluate_v5_baseline(release: str = DEFAULT_RELEASE) -> dict[str, Any]:
    """Evaluate the unchanged promoted parent on the exact V5 gate datasets."""
    _validate_parent_checkpoint()
    release_manifest = _install_release(release)
    release_directory = REMOTE_PUBLIC_ROOT / "releases" / validate_release_name(release)
    output_directory = (
        REMOTE_OUTPUT_ROOT
        / "public-v5-baselines"
        / release
        / EXPECTED_PARENT_MODEL_SHA256[:12]
    )
    output_volume.reload()
    if output_directory.exists():
        cached = _verify_manifest(output_directory)
        if cached.get("parent_model_sha256") != EXPECTED_PARENT_MODEL_SHA256:
            raise ValueError(
                "cached baseline parent hash does not match the expected wave parent"
            )
        return cached
    with tempfile.TemporaryDirectory(prefix="public-v5-baseline-") as temporary:
        local_directory = Path(temporary)
        reports = _evaluate_checkpoint_reports(
            REMOTE_CHECKPOINT,
            "ner_public_v5_seed1.yaml",
            release_directory / "speech_year_dev.jsonl",
            local_directory,
        )
        manifest = _publish_directory(
            local_directory,
            output_directory,
            {
                "data_release": release,
                "data_release_manifest_sha256": _manifest_digest(release_manifest),
                "evaluation_names": sorted(reports),
                "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
            },
        )
    output_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    phase: str = "all",
    seed: str = "both",
    release: str = DEFAULT_RELEASE,
    jobs_release: str = "",
) -> None:
    """Run parallel corpus jobs, deterministic finalization, and parallel GPU seeds.

    ``phase=remix`` rebuilds a release mix from an existing ``jobs_release`` (no
    re-mining) then trains — used to rebalance PER/ORG/LOC vs year-only speech.

    ``phase=uzner-remix`` re-imports UzNER with the current label map into
    ``release``, reuses speech jobs from ``jobs_release``, then trains.
    """
    validate_release_name(release)
    if phase not in {"prepare", "train", "all", "remix", "uzner-remix"}:
        raise ValueError("phase must be prepare, train, all, remix, or uzner-remix")
    if phase in {"remix", "uzner-remix"} and not jobs_release:
        raise ValueError(f"{phase} requires --jobs-release pointing at completed prep jobs")
    result: dict[str, Any] = {"release": release}
    if jobs_release:
        result["jobs_release"] = validate_release_name(jobs_release)
    if phase in {"prepare", "all"}:
        calls = [
            ("uzner-expert", prepare_uzner.spawn("expert", release)),
            ("uzner-temporal", prepare_uzner.spawn("temporal", release)),
            ("common-voice", prepare_speech.spawn("common-voice", release)),
            ("usc", prepare_speech.spawn("usc", release)),
            ("news-youtube", prepare_speech.spawn("news-youtube", release)),
            ("it-youtube", prepare_speech.spawn("it-youtube", release)),
            ("podcasts-tashkent", prepare_speech.spawn("podcasts-tashkent", release)),
            ("uzbekvoice", prepare_speech.spawn("uzbekvoice", release)),
            ("uzbekvoice2", prepare_speech.spawn("uzbekvoice2", release)),
            ("uzbek-news-text", prepare_speech.spawn("uzbek-news-text", release)),
            ("wikipedia-uz", prepare_speech.spawn("wikipedia-uz", release)),
            ("fleurs-uz", prepare_speech.spawn("fleurs-uz", release)),
            ("glotcc-uzn-latn", prepare_speech.spawn("glotcc-uzn-latn", release)),
            ("zero-shot-news", prepare_speech.spawn("zero-shot-news", release)),
            ("oasst2-uz", prepare_speech.spawn("oasst2-uz", release)),
        ]
        result["preparation"] = collect_call_results(calls)
        result["data_release"] = finalize_release.remote(release)
    if phase == "uzner-remix":
        calls = [
            ("uzner-expert", prepare_uzner.spawn("expert", release)),
            ("uzner-temporal", prepare_uzner.spawn("temporal", release)),
        ]
        result["preparation"] = collect_call_results(calls)
        result["data_release"] = finalize_release.remote(release, jobs_release)
    if phase == "remix":
        result["data_release"] = finalize_release.remote(release, jobs_release)
    if phase in {"train", "all", "remix", "uzner-remix"}:
        training_calls = [
            ("promoted-v1-baseline", evaluate_v5_baseline.spawn(release)),
            *[
            (config, train_v5_seed.spawn(config, release))
            for config in selected_config_basenames(seed)
            ],
        ]
        result["evaluation_and_training"] = collect_call_results(training_calls)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
