"""Download and verify local runtime models for a fresh clone.

NER weights ship on a pinned GitHub Release. Whisper Uzbek checkpoints prefetch
from Hugging Face into the configured model cache, matching first-run STT load.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config, resolve_project_path
from .stt.base import ModelLoadError

LOGGER = logging.getLogger(__name__)

NER_REQUIRED_FILES = (
    "config.json",
    "labels.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def downloads_enabled() -> bool:
    """Return False when operators explicitly disable network model bootstrap."""
    raw = os.getenv("SKIP_MODEL_DOWNLOAD") or os.getenv("NER_DOWNLOAD_DISABLED")
    if raw is None:
        return True
    return raw.strip().casefold() not in {"1", "true", "yes", "on"}


def ner_bundle_ready(model_path: Path) -> bool:
    """True when the local NER inference directory has the required files."""
    return model_path.is_dir() and all((model_path / name).is_file() for name in NER_REQUIRED_FILES)


def _required_string(values: Mapping[str, Any], name: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"NER download field {name} must be a non-empty string.")
    return value.strip()


def _download_section(config: AppConfig) -> Mapping[str, Any] | None:
    ner = config.section("ner")
    section = ner.get("download")
    if section is None:
        return None
    if not isinstance(section, Mapping):
        raise ValueError("ner.download must be a mapping when present.")
    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("ner.download.enabled must be boolean.")
    if not enabled:
        return None
    return section


def _asset_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_url(url: str, destination: Path) -> None:
    LOGGER.info("Downloading %s", url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            total = response.headers.get("Content-Length")
            expected = int(total) if total and total.isdigit() else None
            written = 0
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if expected and written % (16 * 1024 * 1024) < len(chunk):
                        LOGGER.info(
                            "NER download progress: %.0f%%",
                            100.0 * written / expected,
                        )
    except urllib.error.URLError as error:
        raise ModelLoadError(f"Could not download NER release asset from {url}.") from error


def download_ner_bundle(config: AppConfig | None = None, *, force: bool = False) -> Path:
    """Ensure `ner.model_path` contains the pinned GitHub Release inference zip."""
    app_config = config or load_config()
    ner = app_config.section("ner")
    configured_path = ner.get("model_path")
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise ValueError("ner.model_path must be a non-empty string.")
    model_path = resolve_project_path(configured_path)
    if ner_bundle_ready(model_path) and not force:
        LOGGER.info("NER bundle already present at %s", model_path)
        return model_path

    section = _download_section(app_config)
    if section is None:
        raise ModelLoadError(
            f"NER model missing at {model_path} and ner.download is disabled or unset."
        )
    if not downloads_enabled():
        raise ModelLoadError(
            f"NER model missing at {model_path} and SKIP_MODEL_DOWNLOAD is set."
        )

    repo = _required_string(section, "github_repo")
    tag = _required_string(section, "release_tag")
    asset = _required_string(section, "asset")
    expected_sha = _required_string(section, "sha256").casefold()
    url = _asset_url(repo, tag, asset)

    with tempfile.TemporaryDirectory(prefix="uzbek-ner-download-") as temp_name:
        temp_root = Path(temp_name)
        archive = temp_root / asset
        extract_root = temp_root / "extract"
        extract_root.mkdir()
        _download_url(url, archive)
        actual_sha = _sha256_file(archive)
        if actual_sha.casefold() != expected_sha:
            raise ModelLoadError(
                "NER release asset SHA-256 mismatch "
                f"(expected {expected_sha}, got {actual_sha})."
            )
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(extract_root)
        if not ner_bundle_ready(extract_root):
            nested = next(
                (
                    path
                    for path in extract_root.rglob("model.safetensors")
                    if ner_bundle_ready(path.parent)
                ),
                None,
            )
            if nested is None:
                raise ModelLoadError("NER release zip is missing required inference files.")
            extract_root = nested.parent
        if model_path.exists():
            shutil.rmtree(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(extract_root, model_path)

    if not ner_bundle_ready(model_path):
        raise ModelLoadError(f"NER install failed under {model_path}.")
    LOGGER.info("Installed NER bundle at %s", model_path)
    return model_path


def ensure_ner_model(
    model_path: Path,
    config: AppConfig | None = None,
    *,
    local_files_only: bool = False,
) -> Path:
    """Return a ready NER path, downloading into the configured path when allowed."""
    app_config = config or load_config()
    resolved = Path(model_path)
    if ner_bundle_ready(resolved):
        return resolved
    if local_files_only or not downloads_enabled():
        raise ModelLoadError(f"NER model is not available at {resolved}.")
    configured = resolve_project_path(str(app_config.section("ner")["model_path"]))
    if resolved.resolve() != configured.resolve():
        raise ModelLoadError(
            f"NER model is not available at {resolved}. "
            "Auto-download only fills the configured ner.model_path."
        )
    return download_ner_bundle(app_config)


def prefetch_stt_models(config: AppConfig | None = None) -> list[str]:
    """Prefetch configured Whisper Uzbek models into the local HF cache."""
    if not downloads_enabled():
        LOGGER.info("Skipping STT prefetch because SKIP_MODEL_DOWNLOAD is set.")
        return []

    from huggingface_hub import snapshot_download

    app_config = config or load_config()
    stt = app_config.section("stt")
    model_ids: list[str] = []
    for key in ("model_id", "fallback_model_id"):
        value = stt.get(key)
        if isinstance(value, str) and value.strip():
            model_ids.append(value.strip())
    if not model_ids:
        raise ValueError("stt.model_id must be configured to prefetch Whisper models.")

    cache_value = os.getenv("MODEL_CACHE_DIR", "./models/cache")
    if not cache_value.strip():
        raise ValueError("MODEL_CACHE_DIR must not be empty.")
    cache_dir = resolve_project_path(cache_value)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fetched: list[str] = []
    for model_id in model_ids:
        LOGGER.info("Prefetching STT model %s into %s", model_id, cache_dir)
        snapshot_download(repo_id=model_id, cache_dir=str(cache_dir))
        fetched.append(model_id)
    return fetched


def ffmpeg_status() -> tuple[bool, str]:
    """Report whether ffmpeg is on PATH and how to install it on this OS."""
    found = shutil.which("ffmpeg")
    if found:
        return True, found
    if sys.platform == "darwin":
        hint = "Install FFmpeg with: brew install ffmpeg"
    elif sys.platform.startswith("win"):
        hint = (
            "Install FFmpeg with: winget install --id Gyan.FFmpeg -e "
            "(or choco install ffmpeg), then reopen the terminal."
        )
    else:
        hint = "Install FFmpeg with your package manager (e.g. apt install ffmpeg)."
    return False, hint


def ensure_runtime_models(
    config: AppConfig | None = None,
    *,
    local_files_only: bool = False,
    force_ner: bool = False,
    prefetch_stt: bool = True,
) -> dict[str, Any]:
    """Download NER (and optionally prefetch STT) for a usable local runtime."""
    app_config = config or load_config()
    ffmpeg_ok, ffmpeg_detail = ffmpeg_status()
    ner_path = resolve_project_path(str(app_config.section("ner")["model_path"]))
    if force_ner or not ner_bundle_ready(ner_path):
        if local_files_only:
            raise ModelLoadError(f"NER model is not available at {ner_path}.")
        ner_path = download_ner_bundle(app_config, force=force_ner)
    stt_models: list[str] = []
    if prefetch_stt and not local_files_only:
        stt_models = prefetch_stt_models(app_config)
    return {
        "ner_path": str(ner_path),
        "stt_models": stt_models,
        "ffmpeg_available": ffmpeg_ok,
        "ffmpeg_detail": ffmpeg_detail,
    }
