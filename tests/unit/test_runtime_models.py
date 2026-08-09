from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from uzbek_speech_entities.config import AppConfig
from uzbek_speech_entities.runtime_models import (
    download_ner_bundle,
    downloads_enabled,
    ensure_ner_model,
    ner_bundle_ready,
    prefetch_stt_models,
)
from uzbek_speech_entities.stt.base import ModelLoadError


def _freeze(values: dict) -> AppConfig:
    from uzbek_speech_entities.config import _freeze as freeze

    return AppConfig(path=Path("configs/app.yaml"), values=freeze(values))


def _config(tmp_path: Path, *, sha256: str, asset: str = "ner-final.zip") -> AppConfig:
    return _freeze(
        {
            "ner": {
                "model_path": str(tmp_path / "models" / "ner" / "final"),
                "download": {
                    "enabled": True,
                    "github_repo": "doston3333/uzbek-speech-entities",
                    "release_tag": "runtime-models-v1",
                    "asset": asset,
                    "sha256": sha256,
                },
            }
        }
    )


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _bundle_files(weights: bytes) -> dict[str, bytes]:
    return {
        "config.json": b"{}",
        "labels.json": b"{}",
        "model.safetensors": weights,
        "special_tokens_map.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "vocab.txt": b"tok",
    }


def _write_bundle(path: Path, weights: bytes) -> None:
    path.mkdir(parents=True)
    for name, contents in _bundle_files(weights).items():
        (path / name).write_bytes(contents)


def _download_payload(monkeypatch: pytest.MonkeyPatch, payload: bytes) -> None:
    monkeypatch.setattr(
        "uzbek_speech_entities.runtime_models._download_url",
        lambda _url, destination: destination.write_bytes(payload),
    )


def test_ner_bundle_ready_requires_all_inference_files(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    assert not ner_bundle_ready(root)
    for name in (
        "config.json",
        "labels.json",
        "model.safetensors",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
    ):
        (root / name).write_bytes(b"x")
    assert ner_bundle_ready(root)


def test_download_ner_bundle_verifies_sha_and_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        "config.json": b"{}",
        "labels.json": b"{}",
        "model.safetensors": b"weights",
        "special_tokens_map.json": b"{}",
        "tokenizer.json": b"{}",
        "tokenizer_config.json": b"{}",
        "vocab.txt": b"tok",
    }
    payload = _zip_bytes(files)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    config = _config(tmp_path, sha256=digest)

    class Response:
        def __init__(self) -> None:
            self.headers = {"Content-Length": str(len(payload))}

        def read(self, size: int = -1) -> bytes:
            data = self._data
            self._data = b""
            return data

        def __enter__(self) -> Response:
            self._data = payload
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "uzbek_speech_entities.runtime_models.urllib.request.urlopen",
        lambda *args, **kwargs: Response(),
    )
    installed = download_ner_bundle(config)
    assert ner_bundle_ready(installed)
    assert (installed / "model.safetensors").read_bytes() == b"weights"


def test_ner_install_copy_failure_preserves_existing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip_bytes(_bundle_files(b"new-weights"))
    import hashlib

    config = _config(tmp_path, sha256=hashlib.sha256(payload).hexdigest())
    destination = Path(config.section("ner")["model_path"])
    _write_bundle(destination, b"old-weights")
    original_files = {path.name: path.read_bytes() for path in destination.iterdir()}
    _download_payload(monkeypatch, payload)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("staging copy failed")

    monkeypatch.setattr("uzbek_speech_entities.runtime_models.shutil.copytree", fail_copy)

    with pytest.raises(OSError, match="staging copy failed"):
        download_ner_bundle(config, force=True)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original_files
    assert not list(destination.parent.glob(".final.staging-*"))


def test_ner_install_activation_failure_restores_existing_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip_bytes(_bundle_files(b"new-weights"))
    import hashlib

    config = _config(tmp_path, sha256=hashlib.sha256(payload).hexdigest())
    destination = Path(config.section("ner")["model_path"])
    _write_bundle(destination, b"old-weights")
    _download_payload(monkeypatch, payload)
    original_rename = Path.rename

    def fail_staging_activation(path: Path, target: str | Path) -> Path:
        if path.name.startswith(".final.staging-") and Path(target) == destination:
            raise OSError("activation failed")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_staging_activation)

    with pytest.raises(ModelLoadError, match="Could not activate"):
        download_ner_bundle(config, force=True)

    assert (destination / "model.safetensors").read_bytes() == b"old-weights"
    assert not list(destination.parent.glob(".final.staging-*"))
    assert not list(destination.parent.glob(".final.backup-*"))


def test_ner_install_replaces_existing_bundle_without_staging_leftovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip_bytes(_bundle_files(b"new-weights"))
    import hashlib

    config = _config(tmp_path, sha256=hashlib.sha256(payload).hexdigest())
    destination = Path(config.section("ner")["model_path"])
    _write_bundle(destination, b"old-weights")
    _download_payload(monkeypatch, payload)

    assert download_ner_bundle(config, force=True) == destination
    assert (destination / "model.safetensors").read_bytes() == b"new-weights"
    assert not list(destination.parent.glob(".final.staging-*"))
    assert not list(destination.parent.glob(".final.backup-*"))


def test_download_ner_bundle_rejects_sha_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip_bytes({"model.safetensors": b"x"})
    config = _config(tmp_path, sha256="0" * 64)

    class Response:
        headers = {"Content-Length": str(len(payload))}

        def read(self, size: int = -1) -> bytes:
            data = getattr(self, "_data", payload)
            self._data = b""
            return data

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        "uzbek_speech_entities.runtime_models.urllib.request.urlopen",
        lambda *args, **kwargs: Response(),
    )
    with pytest.raises(ModelLoadError, match="SHA-256"):
        download_ner_bundle(config)


def test_ensure_ner_model_refuses_foreign_override_paths(tmp_path: Path) -> None:
    config = _config(tmp_path, sha256="abc")
    foreign = tmp_path / "other"
    with pytest.raises(ModelLoadError, match="configured ner.model_path"):
        ensure_ner_model(foreign, config)


@pytest.mark.parametrize(
    ("skip_model_download", "ner_download_disabled", "expected"),
    [
        (None, None, True),
        ("1", None, False),
        (None, "yes", False),
        ("0", "1", False),
        ("false", "unexpected", True),
    ],
)
def test_downloads_enabled_checks_both_disable_flags(
    monkeypatch: pytest.MonkeyPatch,
    skip_model_download: str | None,
    ner_download_disabled: str | None,
    expected: bool,
) -> None:
    for name, value in (
        ("SKIP_MODEL_DOWNLOAD", skip_model_download),
        ("NER_DOWNLOAD_DISABLED", ner_download_disabled),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert downloads_enabled() is expected


def test_prefetch_stt_models_uses_configured_immutable_revisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _freeze(
        {
            "stt": {
                "model_id": "small",
                "model_revision": "0123456789abcdef0123456789abcdef01234567",
                "fallback_model_id": "base",
                "fallback_model_revision": "76543210fedcba9876543210fedcba9876543210",
            }
        }
    )
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "cache"))
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(kwargs),
    )

    assert prefetch_stt_models(config) == ["small", "base"]
    assert [(call["repo_id"], call["revision"]) for call in calls] == [
        ("small", "0123456789abcdef0123456789abcdef01234567"),
        ("base", "76543210fedcba9876543210fedcba9876543210"),
    ]
