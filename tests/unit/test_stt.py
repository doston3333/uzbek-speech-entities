# ruff: noqa: E501, UP037

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uzbek_speech_entities.audio.loader import AudioDecodingError, DecodedAudio
from uzbek_speech_entities.config import AppConfig, project_root
from uzbek_speech_entities.stt.base import ModelLoadError, SpeechToTextService, TranscriptionError
from uzbek_speech_entities.stt.factory import create_stt_service
from uzbek_speech_entities.stt.transformers_backend import (
    TransformersSpeechToTextService,
    _split_audio_on_quiet_boundaries,
)

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _decoded(samples: np.ndarray | None = None, sample_rate: int = 16_000) -> DecodedAudio:
    return DecodedAudio(
        np.asarray(samples if samples is not None else [0.0, 0.2], dtype=np.float32), sample_rate
    )


class _FakeProcessor:
    feature_extractor = SimpleNamespace(chunk_length=30.0)
    calls: list[dict[str, object]] = []
    process_calls: list[tuple[object, dict[str, object]]] = []
    decode_calls: list[tuple[object, dict[str, object]]] = []
    decoded_output: list[object] = ["  Salom Oʻzbekiston  "]

    @classmethod
    def from_pretrained(cls, _model_id: str, **kwargs: object) -> "_FakeProcessor":
        cls.calls.append(kwargs)
        return cls()

    def __call__(self, samples: object, **kwargs: object) -> "_FakeInputs":
        type(self).process_calls.append((samples, kwargs))
        return _FakeInputs(input_features="features", attention_mask="mask")

    def batch_decode(self, generated_ids: object, **kwargs: object) -> list[object]:
        type(self).decode_calls.append((generated_ids, kwargs))
        return type(self).decoded_output


class _FakeInputs(dict[str, object]):
    to_calls: list[tuple[str, object]] = []

    def to(self, device: str, dtype: object) -> "_FakeInputs":
        type(self).to_calls.append((device, dtype))
        return self


class _FakeModel:
    calls: list[dict[str, object]] = []
    devices: list[str] = []
    generate_calls: list[dict[str, object]] = []
    eval_calls = 0

    @classmethod
    def from_pretrained(cls, _model_id: str, **kwargs: object) -> "_FakeModel":
        cls.calls.append(kwargs)
        return cls()

    def to(self, device: str) -> "_FakeModel":
        type(self).devices.append(device)
        return self

    def eval(self) -> "_FakeModel":
        type(self).eval_calls += 1
        return self

    def generate(self, **kwargs: object) -> list[list[int]]:
        type(self).generate_calls.append(kwargs)
        return [[1, 2, 3]]


def _dependencies(mps_available: bool) -> tuple[object, ...]:
    torch = SimpleNamespace(
        float32="float32",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)),
        inference_mode=nullcontext,
    )
    return torch, _FakeProcessor, _FakeModel


def _reset_fakes() -> None:
    _FakeProcessor.calls = []
    _FakeProcessor.process_calls = []
    _FakeProcessor.decode_calls = []
    _FakeProcessor.decoded_output = ["  Salom Oʻzbekiston  "]
    _FakeInputs.to_calls = []
    _FakeModel.calls = []
    _FakeModel.devices = []
    _FakeModel.generate_calls = []
    _FakeModel.eval_calls = 0


def _service(**kwargs: object) -> TransformersSpeechToTextService:
    return TransformersSpeechToTextService("model", Path("cache"), revision=REVISION, **kwargs)


def test_protocol_is_runtime_checkable() -> None:
    service = _service()
    assert isinstance(service, SpeechToTextService)
    assert service.model_id == "model"
    assert service.revision == REVISION


def test_model_loads_once_uses_mps_and_passes_explicit_whisper_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reset_fakes()
    service = _service(chunk_length_seconds=36)
    monkeypatch.setattr(service, "_load_dependencies", lambda: _dependencies(True))
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio", lambda _path: _decoded()
    )

    assert service.transcribe(tmp_path / "input.wav") == "Salom Oʻzbekiston"
    assert service.transcribe(tmp_path / "input.wav") == "Salom Oʻzbekiston"

    assert len(_FakeProcessor.calls) == 1
    assert len(_FakeModel.calls) == 1
    assert _FakeProcessor.calls[0]["revision"] == REVISION
    assert _FakeModel.calls[0]["revision"] == REVISION
    assert _FakeModel.devices == ["mps"]
    assert _FakeModel.eval_calls == 1
    assert len(_FakeProcessor.process_calls) == 2
    samples, process_options = _FakeProcessor.process_calls[0]
    assert isinstance(samples, np.ndarray)
    assert process_options == {
        "sampling_rate": 16_000,
        "return_tensors": "pt",
        "truncation": True,
        "padding": "longest",
        "return_attention_mask": True,
    }
    assert _FakeInputs.to_calls == [("mps", "float32"), ("mps", "float32")]
    assert len(_FakeModel.generate_calls) == 2
    assert _FakeModel.generate_calls[0] == {
        "input_features": "features",
        "attention_mask": "mask",
        "language": "uz",
        "task": "transcribe",
        "return_timestamps": False,
        "use_model_defaults": False,
    }
    assert _FakeProcessor.decode_calls[0][1] == {"skip_special_tokens": True}


def test_transcribe_uses_the_validated_language_task_and_batch_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    service = _service(language="configured-language", task="configured-task", batch_size=1)
    monkeypatch.setattr(service, "_load_dependencies", lambda: _dependencies(True))
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio", lambda _path: _decoded()
    )

    service.transcribe(Path("input.wav"))

    assert service.batch_size == 1
    assert _FakeModel.generate_calls[0]["language"] == service.language
    assert _FakeModel.generate_calls[0]["task"] == service.task


def test_audio_longer_than_the_native_chunk_uses_bounded_short_form_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    service = _service(chunk_length_seconds=30)
    monkeypatch.setattr(service, "_load_dependencies", lambda: _dependencies(True))
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio",
        lambda _path: _decoded(np.zeros(37 * 16_000, dtype=np.float32)),
    )

    transcript = service.transcribe(Path("long.wav"))

    assert transcript == "Salom Oʻzbekiston Salom Oʻzbekiston"
    assert len(_FakeProcessor.process_calls) == 2
    assert all(
        isinstance(samples, np.ndarray) and samples.size <= 30 * 16_000
        for samples, _ in _FakeProcessor.process_calls
    )
    assert all(call["return_timestamps"] is False for call in _FakeModel.generate_calls)


def test_quiet_boundary_chunking_prefers_the_earliest_strong_pause() -> None:
    sample_rate = 100
    samples = np.ones(37 * sample_rate, dtype=np.float32)
    samples[15 * sample_rate : 16 * sample_rate] = 0.0

    chunks = _split_audio_on_quiet_boundaries(samples, sample_rate, 30.0)

    assert len(chunks) == 2
    assert 14.9 <= chunks[0].size / sample_rate <= 15.3
    assert all(chunk.size <= 30 * sample_rate for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), samples)


def test_chunking_without_a_strong_pause_still_preserves_and_bounds_audio() -> None:
    sample_rate = 100
    samples = np.ones(37 * sample_rate, dtype=np.float32)

    chunks = _split_audio_on_quiet_boundaries(samples, sample_rate, 30.0)

    assert len(chunks) == 2
    assert all(chunk.size <= 30 * sample_rate for chunk in chunks)
    np.testing.assert_array_equal(np.concatenate(chunks), samples)


def test_cpu_is_selected_with_explicit_warning_when_mps_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    _reset_fakes()
    service = _service()
    monkeypatch.setattr(service, "_load_dependencies", lambda: _dependencies(False))
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio", lambda _path: _decoded()
    )

    assert service.transcribe(tmp_path / "input.wav") == "Salom Oʻzbekiston"
    assert _FakeModel.devices[-1] == "cpu"
    assert "MPS is unavailable" in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_id": "", "cache_dir": Path("cache"), "revision": REVISION},
        {"model_id": "model", "cache_dir": Path("cache"), "revision": "main"},
        {
            "model_id": "model",
            "cache_dir": Path("cache"),
            "revision": REVISION,
            "chunk_length_seconds": 0,
        },
        {"model_id": "model", "cache_dir": Path("cache"), "revision": REVISION, "batch_size": 2},
        {
            "model_id": "model",
            "cache_dir": Path("cache"),
            "revision": REVISION,
            "device_preference": ("cuda",),
        },
    ],
)
def test_backend_rejects_invalid_constructor_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TransformersSpeechToTextService(**kwargs)


def test_model_load_failure_is_typed_and_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    calls = 0

    def fail() -> tuple[object, ...]:
        nonlocal calls
        calls += 1
        raise RuntimeError("internal model path")

    monkeypatch.setattr(service, "_load_dependencies", fail)
    with pytest.raises(ModelLoadError) as error:
        service.transcribe(Path("audio.wav"))
    assert "internal" not in str(error.value)
    with pytest.raises(ModelLoadError):
        service.transcribe(Path("audio.wav"))
    assert calls == 1


@pytest.mark.parametrize(
    "decoded",
    [
        _decoded(sample_rate=8_000),
        _decoded(np.array([[0.0, 0.1]], dtype=np.float32)),
        _decoded(np.empty(0, dtype=np.float32)),
        _decoded(np.array([np.nan], dtype=np.float32)),
    ],
)
def test_transcribe_rejects_noncanonical_or_invalid_audio(
    monkeypatch: pytest.MonkeyPatch, decoded: DecodedAudio
) -> None:
    _reset_fakes()
    service = _service()
    monkeypatch.setattr(service, "_load_dependencies", lambda: _dependencies(True))
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio", lambda _path: decoded
    )

    with pytest.raises(TranscriptionError):
        service.transcribe(Path("input.wav"))


def test_transcribe_preserves_decode_error_and_wraps_bad_pipeline_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_fakes()
    service = _service()
    monkeypatch.setattr(service, "_load_dependencies", lambda: _dependencies(True))
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio",
        lambda _path: (_ for _ in ()).throw(AudioDecodingError("decode")),
    )
    with pytest.raises(AudioDecodingError, match="decode"):
        service.transcribe(Path("input.wav"))

    service = _service()
    monkeypatch.setattr(service, "_ensure_loaded", lambda: None)
    processor = _FakeProcessor()
    _FakeProcessor.decoded_output = [3]
    service._processor = processor  # noqa: SLF001
    service._model = _FakeModel()  # noqa: SLF001
    service._torch = SimpleNamespace(float32="float32", inference_mode=nullcontext)  # noqa: SLF001
    service._device = "mps"  # noqa: SLF001
    monkeypatch.setattr(
        "uzbek_speech_entities.stt.transformers_backend.decode_audio", lambda _path: _decoded()
    )
    with pytest.raises(TranscriptionError, match="invalid transcript"):
        service.transcribe(Path("input.wav"))


def _app_config(stt: dict[str, object]) -> AppConfig:
    return AppConfig(path=project_root() / "configs/app.yaml", values={"stt": stt})


def _stt_mapping() -> dict[str, object]:
    return {
        "model_id": "small",
        "model_revision": REVISION,
        "fallback_model_id": "base",
        "fallback_model_revision": "76543210fedcba9876543210fedcba9876543210",
        "language": "uz",
        "task": "transcribe",
        "chunk_length_seconds": 30,
        "batch_size": 1,
        "device_preference": ("mps", "cpu"),
    }


def test_factory_uses_config_environment_overrides_and_explicit_only_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _app_config(_stt_mapping())
    monkeypatch.setenv("STT_MODEL_ID", "environment-model")
    monkeypatch.setenv("STT_MODEL_REVISION", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setenv("MODEL_CACHE_DIR", "local-cache")

    service = create_stt_service(config)
    assert service.model_id == "environment-model"
    assert service.revision == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    assert service.cache_dir == project_root() / "local-cache"
    monkeypatch.delenv("STT_MODEL_ID")
    monkeypatch.delenv("STT_MODEL_REVISION")
    assert create_stt_service(config).model_id == "small"
    assert create_stt_service(config, use_fallback_model=True).model_id == "base"


def test_factory_requires_a_revision_for_model_override_and_accepts_revision_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _app_config(_stt_mapping())
    monkeypatch.setenv("STT_MODEL_ID", "environment-model")
    with pytest.raises(ValueError, match="requires STT_MODEL_REVISION"):
        create_stt_service(config)

    monkeypatch.delenv("STT_MODEL_ID")
    monkeypatch.setenv("STT_MODEL_REVISION", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    service = create_stt_service(config)
    assert service.model_id == "small"
    assert service.revision == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


@pytest.mark.parametrize("mutator", ["batch", "device", "language", "revision"])
def test_factory_rejects_invalid_stt_configuration(mutator: str) -> None:
    stt = _stt_mapping()
    if mutator == "batch":
        stt["batch_size"] = 2
    elif mutator == "device":
        stt["device_preference"] = ["cuda"]
    elif mutator == "revision":
        stt["model_revision"] = "main"
    else:
        stt["language"] = ""
    with pytest.raises(ValueError):
        create_stt_service(_app_config(stt))
