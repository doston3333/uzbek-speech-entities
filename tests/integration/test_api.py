from __future__ import annotations

import asyncio
import importlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from fakes import FakeNERPredictor, FakeSTTService
from fastapi.testclient import TestClient
from starlette.requests import Request

from uzbek_speech_entities.api.app import create_app
from uzbek_speech_entities.api.audio_request_limit import MULTIPART_ENVELOPE_ALLOWANCE_BYTES
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
            models=AnalysisModels(
                stt="fake-stt",
                stt_revision="0000000000000000000000000000000000000000",
                ner="models/ner/fake",
            ),
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
        assert (
            response.json()["models"]["stt_revision"] == "0000000000000000000000000000000000000000"
        )
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


def test_default_app_starts_unavailable_when_ner_bootstrap_and_load_fail(
    monkeypatch, caplog
) -> None:
    app_module = importlib.import_module("uzbek_speech_entities.api.app")
    stt = FakeSTTService()
    stt._loaded = True  # type: ignore[attr-defined]
    fallback_predictor = FakeNERPredictor()
    fallback_predictor._loaded = False  # type: ignore[attr-defined]
    calls: list[bool] = []

    def fail_ner_load() -> None:
        raise ModelLoadError("private NER load failure")

    def from_config_with_bootstrap_failure(_config, *, local_files_only: bool = False):
        calls.append(local_files_only)
        if not local_files_only:
            raise ModelLoadError("private NER bootstrap failure")
        fallback_predictor.load = fail_ner_load  # type: ignore[method-assign]
        return fallback_predictor

    monkeypatch.setattr(app_module, "create_stt_service", lambda _config: stt)
    monkeypatch.setattr(
        app_module.NERPredictor,
        "from_config",
        from_config_with_bootstrap_failure,
    )

    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        with TestClient(app_module.create_app(config=load_config())) as client:
            health = client.get("/api/health")

    assert calls == [False, True]
    assert health.status_code == 503
    assert "private NER" not in health.text
    assert "NER model bootstrap failed" in caplog.text
    assert "NER model load failed" in caplog.text


def test_text_api_validation_and_public_offsets() -> None:
    with _client() as client:
        success = client.post("/api/analyze-text", json={"text": "Akmal"})
        assert success.status_code == 200
        body = success.json()
        assert body["entities"][0]["label"] == "PER"
        assert body["normalized_transcript"][0:5] == body["entities"][0]["text"]
        assert body["models"] == {
            "stt": None,
            "stt_revision": None,
            "ner": "models/ner/fake",
        }
        assert client.post("/api/analyze-text", json={"text": ""}).status_code == 422
        assert client.post("/api/analyze-text", json={"text": "x" * 20_001}).status_code == 422
        failed = client.post("/api/analyze-text", json={"text": "explode"})
        assert failed.status_code == 500
        assert "private model detail" not in failed.text


def test_health_responds_while_text_inference_is_blocked() -> None:
    inference_started = Event()
    release_inference = Event()

    class BlockingApiAnalyzer(ApiAnalyzer):
        def analyze_text(self, text: str) -> AnalysisResult:
            inference_started.set()
            if not release_inference.wait(timeout=5):
                raise RuntimeError("test inference was not released")
            return super().analyze_text(text)

    stt = FakeSTTService()
    ner = FakeNERPredictor((Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),))
    stt._loaded = True  # type: ignore[attr-defined]
    ner._loaded = True  # type: ignore[attr-defined]
    analyzer = BlockingApiAnalyzer(stt_service=stt, ner_predictor=ner, audio_config=_config())

    with TestClient(create_app(analyzer=analyzer)) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            inference_response = executor.submit(
                client.post, "/api/analyze-text", json={"text": "Akmal"}
            )
            try:
                assert inference_started.wait(timeout=5)
                assert client.get("/api/health").status_code == 200
            finally:
                release_inference.set()
            assert inference_response.result(timeout=5).status_code == 200


def test_audio_api_rejects_bad_uploads_and_sanitizes_failures() -> None:
    with _client() as client:
        valid = client.post(
            "/api/analyze-audio", files={"file": ("valid.wav", b"audio", "audio/wav")}
        )
        assert valid.status_code == 200
        valid_body = valid.json()
        observed_path = client.app.state.analyzer.observed_audio_path
        assert isinstance(observed_path, Path) and not observed_path.exists()
        assert valid_body["models"] == {
            "stt": "fake-stt",
            "stt_revision": "0000000000000000000000000000000000000000",
            "ner": "models/ner/fake",
        }
        assert all(
            entity["label"] in {"PER", "LOC", "ORG", "DATE"} for entity in valid_body["entities"]
        )
        assert all(
            valid_body["normalized_transcript"][entity["start"] : entity["end"]] == entity["text"]
            for entity in valid_body["entities"]
        )
        assert client.post("/api/analyze-audio").status_code == 422
        assert (
            client.post(
                "/api/analyze-audio", files={"file": ("empty.wav", b"", "audio/wav")}
            ).status_code
            == 400
        )
        assert (
            client.post(
                "/api/analyze-audio", files={"file": ("bad.txt", b"abc", "text/plain")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/analyze-audio", files={"file": ("bad.wav", b"abc", "text/plain")}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/analyze-audio", files={"file": ("big.wav", b"123456789", "audio/wav")}
            ).status_code
            == 413
        )
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


def test_audio_body_limit_rejects_declared_oversize_before_multipart_parsing(monkeypatch) -> None:
    async def multipart_must_not_run(*_args, **_kwargs) -> None:
        raise AssertionError("multipart parser ran despite oversized Content-Length")

    monkeypatch.setattr(Request, "_get_form", multipart_must_not_run)
    declared_size = _config().max_upload_bytes + MULTIPART_ENVELOPE_ALLOWANCE_BYTES + 1

    with _client() as client:
        response = client.post(
            "/api/analyze-audio",
            content=b"small-body",
            headers={
                "content-length": str(declared_size),
                "content-type": "multipart/form-data; boundary=unused",
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "upload_too_large"


def test_audio_body_limit_counts_requests_without_content_length() -> None:
    stt = FakeSTTService()
    ner = FakeNERPredictor()
    stt._loaded = True  # type: ignore[attr-defined]
    ner._loaded = True  # type: ignore[attr-defined]
    app = create_app(
        analyzer=ApiAnalyzer(stt_service=stt, ner_predictor=ner, audio_config=_config())
    )
    request_limit = _config().max_upload_bytes + MULTIPART_ENVELOPE_ALLOWANCE_BYTES
    chunk_size = request_limit // 2 + 1
    multipart_prefix = (
        b"--limit-test\r\n"
        b'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
    )
    messages = [
        {
            "type": "http.request",
            "body": multipart_prefix + b"x" * (chunk_size - len(multipart_prefix)),
            "more_body": True,
        },
        {"type": "http.request", "body": b"y" * chunk_size, "more_body": False},
    ]
    response_messages: list[dict[str, object]] = []

    async def receive() -> dict[str, object]:
        return messages.pop(0)

    async def send(message: dict[str, object]) -> None:
        response_messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/analyze-audio",
        "raw_path": b"/api/analyze-audio",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"multipart/form-data; boundary=limit-test")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }

    asyncio.run(app(scope, receive, send))

    assert response_messages[0]["status"] == 413
    assert json.loads(response_messages[1]["body"]) == {
        "error": {
            "code": "upload_too_large",
            "message": "Audio file exceeds the upload limit.",
        }
    }
