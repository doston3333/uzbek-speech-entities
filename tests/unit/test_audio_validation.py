# ruff: noqa: E501

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from uzbek_speech_entities.audio.loader import (
    AudioDecodingError,
    AudioDurationLimitError,
    DecodedAudio,
    _decode_with_ffmpeg,
    decode_audio,
)
from uzbek_speech_entities.audio.preprocessing import prepared_audio, preprocess_audio
from uzbek_speech_entities.audio.validation import (
    AudioValidationConfig,
    AudioValidationError,
    validate_audio,
)
from uzbek_speech_entities.config import load_config


@pytest.fixture
def audio_config() -> AudioValidationConfig:
    return AudioValidationConfig.from_mapping(
        {
            "target_sample_rate": 16_000,
            "mono": True,
            "max_seconds": 1,
            "max_upload_mb": 1,
            "allowed_extensions": ["wav", "MP3", ".flac"],
        }
    )


def _audio(samples: np.ndarray | None = None, sample_rate: int = 16_000) -> DecodedAudio:
    return DecodedAudio(
        samples=np.asarray(samples if samples is not None else [0.0, 0.25], dtype=np.float32),
        sample_rate=sample_rate,
    )


def test_decoded_audio_is_float32_immutable_and_reports_shape_helpers() -> None:
    decoded = _audio(np.ones((4, 2), dtype=np.float64), 8_000)

    assert decoded.samples.dtype == np.float32
    assert decoded.frames == 4
    assert decoded.channels == 2
    assert decoded.is_mono is False
    assert decoded.duration_seconds == pytest.approx(0.0005)
    with pytest.raises(ValueError):
        decoded.samples[0, 0] = 0.0


@pytest.mark.parametrize("sample_rate", [0, -1])
def test_decoded_audio_requires_positive_sample_rate(sample_rate: int) -> None:
    with pytest.raises(ValueError, match="sample rate"):
        _audio(sample_rate=sample_rate)


def test_decode_audio_wraps_decoder_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import soundfile as sf

    monkeypatch.setattr(sf, "read", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("raw")))
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.loader._decode_with_ffmpeg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fallback raw")),
    )

    with pytest.raises(AudioDecodingError, match="could not be decoded") as error:
        decode_audio(tmp_path / "bad.wav")

    assert "raw" not in str(error.value)


def test_decode_audio_falls_back_for_container_formats_and_transposes_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import soundfile as sf

    candidate = tmp_path / "recording.webm"
    candidate.write_bytes(b"container")
    monkeypatch.setattr(
        sf,
        "read",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unsupported container")),
    )
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.loader._decode_with_ffmpeg",
        lambda *_args, **_kwargs: (
            np.array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], dtype=np.float32),
            48_000,
        ),
    )

    decoded = decode_audio(candidate)

    assert decoded.sample_rate == 48_000
    assert decoded.frames == 3
    assert decoded.channels == 2
    np.testing.assert_allclose(
        decoded.samples,
        np.array([[0.0, 0.3], [0.1, 0.4], [0.2, 0.5]], dtype=np.float32),
    )


def test_decode_audio_caps_soundfile_frames_before_rejecting_long_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import soundfile as sf

    candidate = tmp_path / "compressed.flac"
    candidate.write_bytes(b"compressed")
    monkeypatch.setattr(sf, "info", lambda _path: SimpleNamespace(samplerate=4))

    def bounded_read(
        _path: Path, *, dtype: str, always_2d: bool, frames: int
    ) -> tuple[np.ndarray, int]:
        assert dtype == "float32"
        assert always_2d is False
        assert frames == 5
        return np.ones(frames, dtype=np.float32), 4

    monkeypatch.setattr(sf, "read", bounded_read)
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.loader._decode_with_ffmpeg",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not fall back")),
    )

    with pytest.raises(AudioDurationLimitError, match="duration limit"):
        decode_audio(candidate, max_seconds=1)


def test_ffmpeg_fallback_bounds_duration_rate_and_channels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import soundfile as sf

    command: list[str] = []

    def fake_run(arguments: list[str], **_kwargs: object) -> SimpleNamespace:
        command.extend(arguments)
        return SimpleNamespace(stdout=b"bounded-wav")

    monkeypatch.setattr("uzbek_speech_entities.audio.loader.subprocess.run", fake_run)
    monkeypatch.setattr(
        sf,
        "read",
        lambda *_args, **_kwargs: (np.zeros((2, 2), dtype=np.float32), 48_000),
    )

    samples, sample_rate = _decode_with_ffmpeg(tmp_path / "recording.webm", max_seconds=60)

    assert samples.shape == (2, 2)
    assert sample_rate == 48_000
    assert command[command.index("-t") + 1] == "61.000000"
    assert command[command.index("-ar") + 1] == "48000"
    assert command[command.index("-ac") + 1] == "2"


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({}, "target sample rate"),
        ({"target_sample_rate": True}, "target sample rate"),
        (
            {"target_sample_rate": 16_000, "mono": True, "max_seconds": 0, "max_upload_mb": 1, "allowed_extensions": ["wav"]},
            "limits",
        ),
        (
            {"target_sample_rate": 16_000, "mono": True, "max_seconds": 1, "max_upload_mb": 1, "allowed_extensions": []},
            "extensions",
        ),
    ],
)
def test_validation_config_rejects_invalid_mapping(mapping: dict[str, object], message: str) -> None:
    with pytest.raises(AudioValidationError, match=message):
        AudioValidationConfig.from_mapping(mapping)


def test_validation_config_normalizes_extensions_and_is_immutable(audio_config: AudioValidationConfig) -> None:
    assert audio_config.allowed_extensions == frozenset({"wav", "mp3", "flac"})
    with pytest.raises(AttributeError):
        audio_config.allowed_extensions.add("webm")


def test_application_audio_policy_accepts_ogg() -> None:
    configured = AudioValidationConfig.from_mapping(load_config().section("audio"))

    assert "ogg" in configured.allowed_extensions


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "extension", "too_large"])
def test_validate_audio_rejects_file_policy_violations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio_config: AudioValidationConfig, kind: str
) -> None:
    if kind == "missing":
        candidate = tmp_path / "missing.wav"
    elif kind == "directory":
        candidate = tmp_path / "directory.wav"
        candidate.mkdir()
    else:
        candidate = tmp_path / ("sample.txt" if kind == "extension" else "sample.wav")
        candidate.write_bytes(b"x" if kind != "empty" else b"")
        if kind == "too_large":
            audio_config = replace(audio_config, max_upload_bytes=0)

    with pytest.raises(AudioValidationError):
        validate_audio(candidate, audio_config)


def test_validate_audio_decodes_once_and_preserves_typed_decode_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio_config: AudioValidationConfig
) -> None:
    candidate = tmp_path / "SAMPLE.WAV"
    candidate.write_bytes(b"valid")
    calls = 0

    def fake_decode(_path: Path, *, max_seconds: float | None = None) -> DecodedAudio:
        nonlocal calls
        assert max_seconds == audio_config.max_seconds
        calls += 1
        return _audio()

    monkeypatch.setattr("uzbek_speech_entities.audio.validation.decode_audio", fake_decode)
    assert validate_audio(candidate, audio_config).sample_rate == 16_000
    assert calls == 1
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.validation.decode_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AudioDecodingError("safe")),
    )
    with pytest.raises(AudioDecodingError, match="safe"):
        validate_audio(candidate, audio_config)


def test_validate_audio_maps_bounded_duration_failure_to_policy_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio_config: AudioValidationConfig
) -> None:
    candidate = tmp_path / "compressed.flac"
    candidate.write_bytes(b"compressed")
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.validation.decode_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AudioDurationLimitError("internal duration detail")
        ),
    )

    with pytest.raises(AudioValidationError, match="duration limit") as error:
        validate_audio(candidate, audio_config)

    assert "internal" not in str(error.value)


@pytest.mark.parametrize(
    "decoded",
    [
        _audio(np.empty(0, dtype=np.float32)),
        _audio(np.array([np.nan], dtype=np.float32)),
        _audio(np.ones(16_001, dtype=np.float32)),
    ],
)
def test_validate_audio_enforces_decoded_invariants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    audio_config: AudioValidationConfig,
    decoded: DecodedAudio,
) -> None:
    candidate = tmp_path / "sample.wav"
    candidate.write_bytes(b"valid")
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.validation.decode_audio",
        lambda *_args, **_kwargs: decoded,
    )

    with pytest.raises(AudioValidationError):
        validate_audio(candidate, audio_config)


def test_preprocess_downmixes_resamples_scales_and_checks_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded = _audio(np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32), 8_000)
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.preprocessing._resample",
        lambda samples, _old, _new: np.repeat(samples, 2),
    )

    processed = preprocess_audio(decoded, 16_000)

    assert processed.sample_rate == 16_000
    assert processed.samples.dtype == np.float32
    assert processed.samples.ndim == 1
    assert processed.samples.tolist() == pytest.approx([1.0] * 4)
    with pytest.raises(ValueError):
        processed.samples[0] = 0.0


def test_preprocess_rejects_nonfinite_input_and_resampler_output(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AudioValidationError, match="finite"):
        preprocess_audio(_audio(np.array([np.inf], dtype=np.float32)))
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.preprocessing._resample",
        lambda _samples, _old, _new: np.array([np.nan], dtype=np.float32),
    )
    with pytest.raises(AudioValidationError, match="finite"):
        preprocess_audio(_audio(sample_rate=8_000), 16_000)


def test_prepared_audio_reuses_safe_canonical_wav(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audio_config: AudioValidationConfig
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"safe")
    monkeypatch.setattr("uzbek_speech_entities.audio.preprocessing.validate_audio", lambda *_args: _audio())

    with prepared_audio(source, audio_config) as prepared:
        assert prepared == source


@pytest.mark.parametrize("raises", [False, True])
def test_prepared_audio_creates_and_cleans_up_temporary_wav(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    audio_config: AudioValidationConfig,
    raises: bool,
) -> None:
    source = tmp_path / "stereo.mp3"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.preprocessing.validate_audio",
        lambda *_args: _audio(np.array([[0.0, 0.5]], dtype=np.float32)),
    )
    monkeypatch.setattr(
        "uzbek_speech_entities.audio.preprocessing.preprocess_audio",
        lambda *_args: preprocess_audio(_audio()),
    )
    observed: Path | None = None

    with (pytest.raises(RuntimeError) if raises else nullcontext()):
        with prepared_audio(source, audio_config) as prepared:
            observed = prepared
            assert prepared.exists()
            if raises:
                raise RuntimeError("consumer failure")

    assert observed is not None
    assert not observed.exists()
