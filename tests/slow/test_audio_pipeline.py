# ruff: noqa: E501

"""Opt-in verification against a genuine, locally cached Uzbek STT fixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from jiwer import wer

from uzbek_speech_entities.audio.preprocessing import prepared_audio
from uzbek_speech_entities.audio.validation import AudioValidationConfig
from uzbek_speech_entities.config import load_config, project_root
from uzbek_speech_entities.normalization.evaluation import normalize_evaluation
from uzbek_speech_entities.stt.base import SpeechToTextService
from uzbek_speech_entities.stt.factory import create_stt_service


def _fixture_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        pytest.fail(f"{path.name} must contain a JSON object")
    return metadata


def _assert_acceptable_transcript(
    expected: object,
    maximum_wer: object,
    transcript: str,
) -> None:
    if not isinstance(expected, str) or not expected.strip():
        pytest.fail("fixture metadata must provide a non-empty expected_transcript")
    if isinstance(maximum_wer, bool) or not isinstance(maximum_wer, int | float):
        pytest.fail("fixture metadata maximum_wer must be numeric")
    assert transcript
    observed_wer = wer(normalize_evaluation(expected), normalize_evaluation(transcript))
    assert observed_wer <= float(maximum_wer), (
        f"normalized WER {observed_wer:.3f} exceeds fixture threshold {float(maximum_wer):.3f}"
    )


@pytest.fixture(scope="module")
def audio_config() -> AudioValidationConfig:
    return AudioValidationConfig.from_mapping(load_config().section("audio"))


@pytest.fixture(scope="module")
def local_stt_service() -> SpeechToTextService:
    return create_stt_service(load_config(), local_files_only=True)


@pytest.mark.slow
def test_genuine_uzbek_audio_transcription_uses_only_local_model_files(
    audio_config: AudioValidationConfig,
    local_stt_service: SpeechToTextService,
) -> None:
    fixture = project_root() / "tests/fixtures/sample_uzbek.wav"
    metadata = project_root() / "tests/fixtures/sample_uzbek.json"
    if not fixture.is_file() or not metadata.is_file():
        pytest.skip(
            "requires tests/fixtures/sample_uzbek.wav and sample_uzbek.json with expected transcript metadata"
        )
    fixture_data = _fixture_metadata(metadata)
    with prepared_audio(fixture, audio_config) as canonical_path:
        transcript = local_stt_service.transcribe(canonical_path)

    _assert_acceptable_transcript(
        fixture_data.get("expected_transcript"),
        fixture_data.get("maximum_wer", 0.35),
        transcript,
    )


@pytest.mark.slow
def test_distinct_uzbek_clips_over_thirty_seconds_are_chunked_without_content_loss(
    audio_config: AudioValidationConfig,
    local_stt_service: SpeechToTextService,
    tmp_path: Path,
) -> None:
    fixture_paths = [
        project_root() / "tests/fixtures/sample_uzbek.wav",
        project_root() / "tests/fixtures/sample_uzbek_2.wav",
    ]
    metadata_paths = [path.with_suffix(".json") for path in fixture_paths]
    if not all(path.is_file() for path in [*fixture_paths, *metadata_paths]):
        pytest.skip("requires both genuine Uzbek audio fixtures and their metadata")

    metadata = [_fixture_metadata(path) for path in metadata_paths]
    audio_parts: list[np.ndarray] = []
    sample_rate: int | None = None
    for fixture in fixture_paths:
        samples, fixture_rate = sf.read(fixture, dtype="float32", always_2d=False)
        assert samples.ndim == 1
        if sample_rate is None:
            sample_rate = fixture_rate
        assert fixture_rate == sample_rate
        audio_parts.append(np.asarray(samples, dtype=np.float32))
    assert sample_rate == 16_000

    one_second_pause = np.zeros(sample_rate, dtype=np.float32)
    combined = np.concatenate([audio_parts[0], one_second_pause, audio_parts[1]])
    assert combined.size / sample_rate > 30.0
    combined_path = tmp_path / "distinct-uzbek-long.wav"
    sf.write(combined_path, combined, sample_rate, subtype="PCM_16")

    with prepared_audio(combined_path, audio_config) as canonical_path:
        transcript = local_stt_service.transcribe(canonical_path)

    expected = " ".join(str(item["expected_transcript"]) for item in metadata)
    maximum_wer = max(float(item.get("maximum_wer", 0.35)) for item in metadata)
    _assert_acceptable_transcript(expected, maximum_wer, transcript)
