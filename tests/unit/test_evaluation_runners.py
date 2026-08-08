from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeNERPredictor, FakeSTTService

from evaluation.benchmark_runtime import RuntimeMeasurement, measure_runtime
from evaluation.config import load_evaluation_config
from evaluation.create_error_report import (
    build_error_records,
    classify_ner_errors,
    classify_normalization_errors,
    classify_stt_errors,
    run_report,
)
from evaluation.dataset import EvaluationDataset, EvaluationSample, GoldEntity
from evaluation.evaluate_pipeline import (
    ABLATIONS,
    _prediction_index,
    evaluate_ablation,
)
from evaluation.evaluate_pipeline import (
    run_evaluation as run_pipeline_evaluation,
)
from evaluation.evaluate_stt import evaluate_stt_service
from uzbek_speech_entities.audio.validation import AudioValidationConfig
from uzbek_speech_entities.config import load_config, project_root
from uzbek_speech_entities.ner.schemas import Entity


def _dataset() -> EvaluationDataset:
    transcript = "Akmal Toshkentga bordi."
    return EvaluationDataset(
        metadata_path=Path("metadata.jsonl"),
        samples=(
            EvaluationSample(
                id="audio-001",
                file=project_root() / "tests/fixtures/sample_audio.wav",
                gold_transcript=transcript,
                entities=(
                    GoldEntity("Akmal", "PER", 0, 5),
                    GoldEntity("Toshkentga", "LOC", 6, 16),
                ),
                speaker_id="speaker-01",
                conditions=("quiet",),
            ),
        ),
    )


def test_stt_service_runner_reports_transcript_mentions_and_runtime() -> None:
    dataset = _dataset()
    service = FakeSTTService(dataset.samples[0].gold_transcript)
    audio_config = AudioValidationConfig.from_mapping(load_config().section("audio"))
    result = evaluate_stt_service(
        model_key="base",
        service=service,
        dataset=dataset,
        audio_config=audio_config,
        resolved_revision="test-revision",
    )

    assert result.summary["raw_wer"] == 0
    assert result.summary["normalized_wer"] == 0
    assert result.summary["PER_mention_accuracy"] == 1
    assert result.summary["LOC_mention_accuracy"] == 1
    assert result.summary["ORG_mention_accuracy"] is None
    assert result.summary["real_time_factor"] >= 0
    assert result.predictions[0]["resolved_revision"] == "test-revision"


def test_required_ablation_matrix_and_gold_span_evaluation() -> None:
    assert [item.id for item in ABLATIONS] == list("ABCDEFGH")
    assert [(item.stt_model, item.normalization, item.ner_run) for item in ABLATIONS] == [
        ("base", False, "clean"),
        ("base", True, "clean"),
        ("base", True, "augmented"),
        ("small", False, "clean"),
        ("small", True, "clean"),
        ("small", True, "augmented"),
        (None, True, "clean"),
        (None, True, "augmented"),
    ]
    predictor = FakeNERPredictor(
        (
            Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),
            Entity(text="Toshkentga", label="LOC", start=6, end=16, score=0.9),
        )
    )
    load_measurement = RuntimeMeasurement(
        value=None,
        elapsed_seconds=0.1,
        starting_rss_mb=10,
        peak_rss_mb=12,
        peak_rss_delta_mb=2,
    )
    result = evaluate_ablation(
        ablation=ABLATIONS[6],
        predictor=predictor,
        dataset=_dataset(),
        stt_predictions={},
        load_measurement=load_measurement,
    )

    assert result.summary["ablation"] == "G"
    assert result.summary["overall_exact_span_f1"] == 1
    assert result.summary["four_class_macro_f1"] == 0.5
    assert result.predictions[0]["match_mode"] == "span"


def test_stt_prediction_index_rejects_wrong_model_provenance() -> None:
    dataset = _dataset()
    rows = [
        {
            "model_key": model,
            "model_id": f"expected/{model}" if model == "base" else "wrong/small",
            "resolved_revision": "revision-1",
            "sample_id": "audio-001",
            "raw_transcript": dataset.samples[0].gold_transcript,
        }
        for model in ("base", "small")
    ]

    with pytest.raises(ValueError, match="wrong model_id"):
        _prediction_index(
            rows,
            dataset,
            {"base": "expected/base", "small": "expected/small"},
        )


def test_audio_ablation_reports_aligned_exact_span_f1() -> None:
    dataset = _dataset()
    predictor = FakeNERPredictor(
        (
            Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),
            Entity(text="Toshkentga", label="LOC", start=6, end=16, score=0.9),
        )
    )
    load_measurement = RuntimeMeasurement(
        value=None,
        elapsed_seconds=0.1,
        starting_rss_mb=10,
        peak_rss_mb=12,
        peak_rss_delta_mb=2,
    )
    stt_rows = {
        ("base", "audio-001"): {"raw_transcript": dataset.samples[0].gold_transcript}
    }

    result = evaluate_ablation(
        ablation=ABLATIONS[1],
        predictor=predictor,
        dataset=dataset,
        stt_predictions=stt_rows,
        load_measurement=load_measurement,
    )

    assert result.summary["overall_exact_span_f1"] == 1
    assert result.predictions[0]["span_alignment"] == "exact-token-alignment"


def test_pipeline_runner_emits_all_eight_rows_with_fake_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset()
    config = load_evaluation_config(project_root() / "configs/evaluation_smoke.yaml")
    compliance = SimpleNamespace(compliant=True, issues=())
    stt_rows = [
        {
            "model_key": model,
            "model_id": config.stt_model_ids[model],
            "resolved_revision": f"{model}-revision",
            "sample_id": "audio-001",
            "raw_transcript": dataset.samples[0].gold_transcript,
        }
        for model in ("base", "small")
    ]
    predictor = FakeNERPredictor(
        (
            Entity(text="Akmal", label="PER", start=0, end=5, score=0.9),
            Entity(text="Toshkentga", label="LOC", start=6, end=16, score=0.9),
        )
    )
    monkeypatch.setattr(
        "evaluation.evaluate_pipeline.load_dataset", lambda *_args, **_kwargs: dataset
    )
    monkeypatch.setattr(
        "evaluation.evaluate_pipeline.build_compliance_report",
        lambda *_args, **_kwargs: compliance,
    )
    monkeypatch.setattr("evaluation.evaluate_pipeline.write_report", lambda *_args: None)
    monkeypatch.setattr("evaluation.evaluate_pipeline.read_jsonl", lambda *_args: stt_rows)
    monkeypatch.setattr(
        "evaluation.evaluate_pipeline.load_run",
        lambda name, _path: SimpleNamespace(checkpoint=Path(f"models/ner/{name}/checkpoint")),
    )
    monkeypatch.setattr(
        "evaluation.evaluate_pipeline._build_ner_predictor",
        lambda *_args, **_kwargs: predictor,
    )
    monkeypatch.setattr(
        "evaluation.evaluate_pipeline._release_accelerator_memory", lambda: None
    )
    monkeypatch.setattr("evaluation.evaluate_pipeline.write_csv_atomic", lambda *_args: None)
    monkeypatch.setattr("evaluation.evaluate_pipeline.write_jsonl_atomic", lambda *_args: None)
    monkeypatch.setattr("evaluation.evaluate_pipeline.write_json_atomic", lambda *_args: None)

    summaries = run_pipeline_evaluation(config, allow_incomplete_dataset=False)

    assert [row["ablation"] for row in summaries] == list("ABCDEFGH")
    assert all(isinstance(row["overall_exact_span_f1"], float) for row in summaries)


def test_error_taxonomies_and_required_detailed_columns() -> None:
    gold = [
        {"text": "Akmal", "label": "PER", "start": 0, "end": 5},
        {"text": "Toshkent", "label": "LOC", "start": 6, "end": 14},
    ]
    predicted = [
        {"text": "Akmal", "label": "LOC", "start": 0, "end": 5, "score": 0.9}
    ]
    stt_errors = classify_stt_errors(
        "Akmal Toshkent bordi", "Akmal bordi bordi", gold, ["background_noise"]
    )
    assert "Location mistranscribed" in stt_errors
    assert "Repetition" in stt_errors
    assert "Noise-related error" in stt_errors
    ner_errors = classify_ner_errors(gold, predicted)
    assert "Person classified as location" in ner_errors
    assert "Missed location" in ner_errors
    normalization_errors = classify_normalization_errors(
        "Akmal", "Akmal", [{"text": "wrong", "label": "PER", "start": 0, "end": 5}]
    )
    assert normalization_errors == ("Offset mismatch",)

    rows = [
        {
            "ablation": "E",
            "sample_id": "audio-001",
            "gold_transcript": "Akmal Toshkent bordi",
            "raw_transcript": "Akmal bordi bordi",
            "normalized_transcript": "Akmal bordi bordi",
            "gold_entities": gold,
            "predicted_entities": predicted,
            "conditions": ["background_noise"],
        }
    ]
    records = build_error_records(rows, primary_ablation="E")
    assert set(records[0]) == {
        "audio_id",
        "gold_transcript",
        "raw_transcript",
        "normalized_transcript",
        "gold_entities",
        "predicted_entities",
        "stt_error",
        "ner_error",
        "normalization_error",
        "notes",
    }


def test_error_report_rejects_noncompliant_dataset_without_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_evaluation_config(project_root() / "configs/evaluation_smoke.yaml")
    compliance = type(
        "Compliance",
        (),
        {"compliant": False, "issues": ("too few recordings",)},
    )()
    monkeypatch.setattr(
        "evaluation.create_error_report.load_dataset", lambda *_args, **_kwargs: _dataset()
    )
    monkeypatch.setattr(
        "evaluation.create_error_report.build_compliance_report",
        lambda *_args, **_kwargs: compliance,
    )
    monkeypatch.setattr("evaluation.create_error_report.write_report", lambda *_args: None)

    with pytest.raises(ValueError, match="not Phase 8 compliant"):
        run_report(config)


def test_runtime_measurement_and_default_config_are_explicit() -> None:
    measurement = measure_runtime(lambda: "done", sample_interval_seconds=0.001)
    assert measurement.value == "done"
    assert measurement.elapsed_seconds >= 0
    assert measurement.peak_rss_mb >= measurement.starting_rss_mb
    config = load_evaluation_config()
    assert config.stt_model_ids == {
        "base": "navai-uz/whisper-base-uzbek",
        "small": "navai-uz/whisper-small-uzbek",
    }
    assert config.require_compliance
    assert "data/private_test/results" in str(config.stt_predictions_path)
