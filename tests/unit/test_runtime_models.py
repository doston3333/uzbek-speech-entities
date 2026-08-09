from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from uzbek_speech_entities.config import AppConfig
from uzbek_speech_entities.runtime_models import (
    download_ner_bundle,
    ensure_ner_model,
    ner_bundle_ready,
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
