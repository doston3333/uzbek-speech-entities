"""Continue fine-tuning the promoted speech NER model on a Modal L4 GPU."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import modal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace")
REMOTE_CONFIG_DIR = REMOTE_ROOT / "configs"
REMOTE_CHECKPOINT = REMOTE_ROOT / "models" / "ner" / "final"
REMOTE_OUTPUT_ROOT = Path("/outputs")
VOLUME_NAME = "uzbek-speech-ner-continuation-runs"
EXPECTED_PARENT_MODEL_SHA256 = "af2993de66f7a36b4ff5c8b6bd68e08f04183e56e9ea160a745238f3d06ed2a0"
INFERENCE_CHECKPOINT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    }
)
APPROVED_CONFIGS: dict[str, tuple[int, str]] = {
    "ner_speech_continuation_v4_seed1.yaml": (
        20260811,
        "speech-continuation-20260807-v4-seed1",
    ),
    "ner_speech_continuation_v4_seed2.yaml": (
        20260812,
        "speech-continuation-20260807-v4-seed2",
    ),
}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_ROOT / "requirements-modal.txt"))
    .env({"PYTHONPATH": ":".join((str(REMOTE_ROOT / "training"), str(REMOTE_ROOT / "src")))})
    .add_local_dir(PROJECT_ROOT / "src", remote_path=str(REMOTE_ROOT / "src"))
    .add_local_dir(PROJECT_ROOT / "training", remote_path=str(REMOTE_ROOT / "training"))
    .add_local_dir(PROJECT_ROOT / "configs", remote_path=str(REMOTE_CONFIG_DIR))
    .add_local_dir(
        PROJECT_ROOT / "data" / "processed" / "ner",
        remote_path=str(REMOTE_ROOT / "data" / "processed" / "ner"),
    )
)
for checkpoint_file in sorted(INFERENCE_CHECKPOINT_FILES):
    image = image.add_local_file(
        PROJECT_ROOT / "models" / "ner" / "final" / checkpoint_file,
        remote_path=str(REMOTE_CHECKPOINT / checkpoint_file),
    )
app = modal.App("uzbek-speech-ner-continuation")
output_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)


def selected_config_basenames(seed: str) -> tuple[str, ...]:
    """Select one approved seed or both in their deterministic order."""
    selected = {
        "seed1": ("ner_speech_continuation_v4_seed1.yaml",),
        "seed2": ("ner_speech_continuation_v4_seed2.yaml",),
        "both": tuple(APPROVED_CONFIGS),
    }
    try:
        return selected[seed]
    except KeyError as error:
        raise ValueError("seed must be one of: seed1, seed2, both") from error


def rewritten_remote_config(config_basename: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Rewrite only checkpoint and output paths for an approved remote run."""
    if Path(config_basename).name != config_basename or config_basename not in APPROVED_CONFIGS:
        raise ValueError("only the two approved continuation configs may run on Modal")
    rewritten = copy.deepcopy(dict(values))
    model = rewritten.get("model")
    output = rewritten.get("output")
    if not isinstance(model, dict) or not isinstance(output, dict):
        raise ValueError("approved config is missing model or output sections")
    _, output_name = APPROVED_CONFIGS[config_basename]
    model["checkpoint"] = str(REMOTE_CHECKPOINT)
    output["directory"] = str(REMOTE_OUTPUT_ROOT / output_name)
    return rewritten


def _write_remote_config(config_basename: str) -> tuple[Path, dict[str, Any]]:
    import yaml

    if Path(config_basename).name != config_basename or config_basename not in APPROVED_CONFIGS:
        raise ValueError("only the two approved continuation configs may run on Modal")
    source_path = REMOTE_CONFIG_DIR / config_basename
    values = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError(f"approved config is not a mapping: {config_basename}")
    rewritten = rewritten_remote_config(config_basename, values)
    path = Path("/tmp") / f"modal-{config_basename}"
    path.write_text(yaml.safe_dump(rewritten, sort_keys=False), encoding="utf-8")
    return path, rewritten


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_parent_checkpoint() -> None:
    parent_model = REMOTE_CHECKPOINT / "model.safetensors"
    actual = _sha256(parent_model)
    if actual != EXPECTED_PARENT_MODEL_SHA256:
        raise RuntimeError(
            f"uploaded parent checkpoint does not match the promoted final NER model: {actual}"
        )


def _stage_inference_bundle(output_directory: Path) -> dict[str, Any]:
    pointer = json.loads((output_directory / "best_checkpoint.json").read_text(encoding="utf-8"))
    relative_checkpoint = pointer.get("checkpoint")
    if not isinstance(relative_checkpoint, str) or not relative_checkpoint:
        raise RuntimeError("training did not write a valid best-checkpoint pointer")
    checkpoint = output_directory / relative_checkpoint
    inference_directory = output_directory / "inference"
    if inference_directory.exists():
        raise FileExistsError(f"refusing to replace inference bundle: {inference_directory}")
    inference_directory.mkdir()
    files: dict[str, dict[str, str | int]] = {}
    for filename in sorted(INFERENCE_CHECKPOINT_FILES):
        source = checkpoint / filename
        if not source.is_file():
            raise FileNotFoundError(f"best checkpoint is missing inference file: {source}")
        destination = inference_directory / filename
        shutil.copyfile(source, destination)
        files[filename] = {"bytes": destination.stat().st_size, "sha256": _sha256(destination)}
    manifest: dict[str, Any] = {
        "files": files,
        "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
        "source_checkpoint": relative_checkpoint,
    }
    (inference_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _summary(config_basename: str, output_directory: Path) -> dict[str, str | int | None]:
    seed, _ = APPROVED_CONFIGS[config_basename]
    best_checkpoint_path = output_directory / "best_checkpoint.json"
    best_checkpoint: str | None = None
    if best_checkpoint_path.is_file():
        payload = json.loads(best_checkpoint_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("checkpoint"), str):
            best_checkpoint = payload["checkpoint"]
    return {
        "best_checkpoint": best_checkpoint,
        "config": config_basename,
        "inference_directory": str(output_directory / "inference"),
        "output_directory": str(output_directory),
        "parent_model_sha256": EXPECTED_PARENT_MODEL_SHA256,
        "seed": seed,
    }


@app.function(
    image=image,
    gpu="L4",
    timeout=60 * 60,
    volumes={str(REMOTE_OUTPUT_ROOT): output_volume},
    include_source=False,
)
def train_continuation_seed(config_basename: str) -> dict[str, str | int | None]:
    """Continue one approved seed from the exact promoted final NER model."""
    _validate_parent_checkpoint()
    config_path, rewritten = _write_remote_config(config_basename)
    output = rewritten["output"]
    assert isinstance(output, dict)
    output_directory = Path(str(output["directory"]))
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
        _stage_inference_bundle(output_directory)
    finally:
        subprocess.run(["sync", str(REMOTE_OUTPUT_ROOT)], check=True)
        output_volume.commit()
    return _summary(config_basename, output_directory)


@app.local_entrypoint()
def main(seed: str = "both") -> None:
    """Run seed1, seed2, or both sequentially from ``modal run``."""
    results = [train_continuation_seed.remote(config) for config in selected_config_basenames(seed)]
    print(json.dumps({"runs": results}, sort_keys=True))
