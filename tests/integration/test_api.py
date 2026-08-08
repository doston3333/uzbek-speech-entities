from __future__ import annotations

import importlib
from pathlib import Path

from fakes import FakeNERPredictor, FakeSTTService
from fastapi.testclient import TestClient

from uzbek_speech_entities.api.app import create_app
from uzbek_speech_entities.audio.loader import AudioDecodingError
from uzbek_speech_entities.audio.validation import AudioValidationConfig
from uzbek_speech_entities.config import load_config
from uzbek_speech_entities.ner.schemas import Entity
from uzbek_speech_entities.ner.spans import NERPredictionError
from uzbek_speech_entities.pipeline.analyzer import SpeechEntityAnalyzer
from uzbek_speech_entities.pipeline.schemas import AnalysisModels, AnalysisResult, Timing
from uzbek_speech_entities.stt.base import ModelLoadError, TranscriptionError


def _config() -> AudioValidationConfig:
    return AudioValidationConfig(16_000, True, 60, 8, frozenset({"wav"}))


class ApiAnalyzer(SpeechEntityAnalyzer):
    observed_audio_path: Path | None = None

    def analyze_text(self, text: str) -> AnalysisResult:
        if text == "explode":
            raise NERPredictionError("private model detail")
        return super().analyze_text(text)

    def analyze_audio(self, audio_path: Path) -> AnalysisResult:
        self.observed_audio_path = audio_path
        payload = audio_path.read_bytes()
        if payload == b"decode":
            raise AudioDecodingError("private decoder detail")
        if payload == b"fail":
            raise TranscriptionError("not exposed")
        return AnalysisResult(
            raw_transcript="Akmal",
            normalized_transcript="Akmal",
            entities=(Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),),
            timing=Timing(
                audio_preprocessing_ms=0,
                stt_ms=0,
                normalization_ms=0,
                ner_ms=0,
                total_ms=0,
            ),
            models=AnalysisModels(stt="fake-stt", ner="models/ner/fake"),
        )


def _client(*, loaded: bool = True) -> TestClient:
    stt = FakeSTTService()
    ner = FakeNERPredictor((Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),))
    stt._loaded = loaded  # type: ignore[attr-defined]
    ner._loaded = loaded  # type: ignore[attr-defined]
    analyzer = ApiAnalyzer(stt_service=stt, ner_predictor=ner, audio_config=_config())
    return TestClient(create_app(analyzer=analyzer))


def test_health_ok_and_model_unavailable() -> None:
    with _client() as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["models"]["ner_loaded"] is True
    with _client(loaded=False) as client:
        assert client.get("/api/health").status_code == 503
        unavailable = client.post("/api/analyze-text", json={"text": "Akmal"})
        assert unavailable.status_code == 503
        unavailable_audio = client.post(
            "/api/analyze-audio", files={"file": ("audio.wav", b"audio", "audio/wav")}
        )
        assert unavailable_audio.status_code == 503


def test_frontend_static_assets_and_security_headers() -> None:
    with _client() as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.headers["content-type"].startswith("text/html")
        assert "default-src 'self'" in root.headers["content-security-policy"]
        assert root.headers["permissions-policy"] == "microphone=(self)"
        assert root.headers["x-content-type-options"] == "nosniff"
        assert root.headers["referrer-policy"] == "no-referrer"
        assert client.get("/assets/styles.css").headers["content-type"].startswith("text/css")
        assert client.get("/assets/app.js").headers["content-type"].startswith("text/javascript")


def test_default_lifespan_keeps_text_available_when_only_stt_load_fails(
    monkeypatch,
) -> None:
    app_module = importlib.import_module("uzbek_speech_entities.api.app")
    stt = FakeSTTService()
    stt._loaded = False  # type: ignore[attr-defined]
    ner = FakeNERPredictor()
    ner._loaded = False  # type: ignore[attr-defined]

    def fail_stt_load() -> None:
        raise ModelLoadError("private model path")

    stt.load = fail_stt_load  # type: ignore[method-assign]
    monkeypatch.setattr(app_module, "create_stt_service", lambda _config: stt)
    monkeypatch.setattr(
        app_module.NERPredictor,
        "from_config",
        lambda _config: ner,
    )

    with TestClient(app_module.create_app(config=load_config())) as client:
        assert client.get("/api/health").status_code == 503
        assert ner.loaded is True
        assert client.post("/api/analyze-text", json={"text": "Akmal"}).status_code == 200


def test_text_api_validation_and_public_offsets() -> None:
    with _client() as client:
        success = client.post("/api/analyze-text", json={"text": "Akmal"})
        assert success.status_code == 200
        body = success.json()
        assert body["entities"][0]["label"] == "PER"
        assert body["normalized_transcript"][0:5] == body["entities"][0]["text"]
        assert body["models"] == {"stt": None, "ner": "models/ner/fake"}
        assert client.post("/api/analyze-text", json={"text": ""}).status_code == 422
        assert client.post("/api/analyze-text", json={"text": "x" * 20_001}).status_code == 422
        failed = client.post("/api/analyze-text", json={"text": "explode"})
        assert failed.status_code == 500
        assert "private model detail" not in failed.text


def test_audio_api_rejects_bad_uploads_and_sanitizes_failures() -> None:
    with _client() as client:
        valid = client.post(
            "/api/analyze-audio", files={"file": ("valid.wav", b"audio", "audio/wav")}
        )
        assert valid.status_code == 200
        valid_body = valid.json()
        observed_path = client.app.state.analyzer.observed_audio_path
        assert isinstance(observed_path, Path) and not observed_path.exists()
        assert valid_body["models"] == {"stt": "fake-stt", "ner": "models/ner/fake"}
        assert all(
            entity["label"] in {"PER", "LOC", "ORG", "DATE"}
            for entity in valid_body["entities"]
        )
        assert all(
            valid_body["normalized_transcript"][entity["start"] : entity["end"]]
            == entity["text"]
            for entity in valid_body["entities"]
        )
        assert client.post("/api/analyze-audio").status_code == 422
        assert client.post(
            "/api/analyze-audio", files={"file": ("empty.wav", b"", "audio/wav")}
        ).status_code == 400
        assert client.post(
            "/api/analyze-audio", files={"file": ("bad.txt", b"abc", "text/plain")}
        ).status_code == 422
        assert client.post(
            "/api/analyze-audio", files={"file": ("bad.wav", b"abc", "text/plain")}
        ).status_code == 422
        assert client.post(
            "/api/analyze-audio", files={"file": ("big.wav", b"123456789", "audio/wav")}
        ).status_code == 413
        invalid = client.post(
            "/api/analyze-audio", files={"file": ("invalid.wav", b"decode", "audio/wav")}
        )
        assert invalid.status_code == 400
        assert "private decoder detail" not in invalid.text
        failed = client.post(
            "/api/analyze-audio", files={"file": ("fail.wav", b"fail", "audio/wav")}
        )
        assert failed.status_code == 500
        assert "not exposed" not in failed.text
