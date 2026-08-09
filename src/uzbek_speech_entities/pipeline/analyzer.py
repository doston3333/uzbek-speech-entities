"""FastAPI-independent orchestration of audio/text analysis."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from time import monotonic

from ..audio.preprocessing import prepared_audio
from ..audio.validation import AudioValidationConfig
from ..ner.offset_tokens import tokenize_words
from ..ner.predictor import NERService
from ..ner.rules.person_gazetteer import gazetteer_boundary_expansion_candidates
from ..ner.rules.temporal import temporal_candidates
from ..ner.schemas import Entity
from ..ner.span_resolver import Candidate
from ..ner.spans import validate_entity_spans
from ..ner.speech_extractor import SpeechNERRescue
from ..normalization.aligned_tokens import AnalysisNormalization
from ..normalization.analysis_normalizer import normalize_speech_analysis
from ..normalization.runtime import normalize_runtime
from ..normalization.span_projection import project_analysis_span
from ..stt.base import ModelLoadError, SpeechToTextService
from .schemas import AnalysisModels, AnalysisResult, Timing

MAX_TEXT_CHARS = 20_000


class TextValidationError(ValueError):
    """Raised when a user-submitted text payload is not accepted."""


class SpeechEntityAnalyzer:
    """Compose injected STT, normalization, and NER without web dependencies."""

    def __init__(
        self,
        *,
        stt_service: SpeechToTextService | None,
        ner_predictor: NERService,
        audio_config: AudioValidationConfig,
        normalizer: Callable[[str], str] = normalize_runtime,
        clock: Callable[[], float] = monotonic,
        speech_rescue_enabled: bool = False,
        speech_rescue: SpeechNERRescue | None = None,
        analysis_normalization_enabled: bool = False,
        normalized_confidence_threshold: float = 0.70,
    ) -> None:
        self.stt_service = stt_service
        self.ner_predictor = ner_predictor
        self.audio_config = audio_config
        self._normalizer = normalizer
        self._clock = clock
        if not isinstance(speech_rescue_enabled, bool):
            raise ValueError("speech_rescue_enabled must be boolean")
        if not isinstance(analysis_normalization_enabled, bool):
            raise ValueError("analysis_normalization_enabled must be boolean")
        if (
            isinstance(normalized_confidence_threshold, bool)
            or not isinstance(normalized_confidence_threshold, int | float)
            or not 0.0 <= normalized_confidence_threshold <= 1.0
        ):
            raise ValueError("normalized_confidence_threshold must be in [0, 1]")
        self._speech_rescue_enabled = speech_rescue_enabled
        self._speech_rescue = speech_rescue or SpeechNERRescue()
        self._analysis_normalization_enabled = analysis_normalization_enabled
        self._normalized_confidence_threshold = float(normalized_confidence_threshold)

    @staticmethod
    def _milliseconds(start: float, end: float) -> float:
        return max(0.0, (end - start) * 1000.0)

    def _models(self, *, include_stt: bool) -> AnalysisModels:
        return AnalysisModels(
            stt=self.stt_service.model_id if include_stt and self.stt_service else None,
            stt_revision=(self.stt_service.revision if include_stt and self.stt_service else None),
            ner=str(self.ner_predictor.model_path),
        )

    @staticmethod
    def _normalized_candidates(
        display_text: str,
        analysis_entities: tuple[Entity, ...],
        analysis_view: AnalysisNormalization,
        confidence_threshold: float,
    ) -> tuple[Candidate, ...]:
        """Admit only transformed, label-compatible analysis model evidence."""
        expansion_evidence = {
            (candidate.label, candidate.start, candidate.end)
            for candidate in gazetteer_boundary_expansion_candidates(tokenize_words(display_text))
        }
        date_boundaries = temporal_candidates(tokenize_words(display_text))
        candidates: list[Candidate] = []
        for entity in analysis_entities:
            if entity.score is None or entity.score < confidence_threshold:
                continue
            projection = project_analysis_span(analysis_view, entity.start, entity.end)
            if projection is None:
                continue
            source_start, source_end = projection
            transformations = {
                token.transformation
                for token in analysis_view.tokens
                if token.analysis_start < entity.end and entity.start < token.analysis_end
            }
            if entity.label == "DATE" and "temporal_itn" in transformations:
                matching_dates = tuple(
                    candidate
                    for candidate in date_boundaries
                    if candidate.start <= source_start
                    and source_end <= candidate.end
                    and source_start < candidate.end
                    and candidate.start < source_end
                )
                if matching_dates:
                    completed = min(
                        matching_dates,
                        key=lambda candidate: (candidate.end - candidate.start, candidate.start),
                    )
                    source_start, source_end = completed.start, completed.end
            supported = (
                (entity.label == "DATE" and "temporal_itn" in transformations)
                or (
                    entity.label == "PER"
                    and bool(transformations & {"person_phrase", "person_name"})
                )
                or (entity.label == "ORG" and "organization_case" in transformations)
                or (entity.label == "LOC" and "location_case" in transformations)
                or (
                    entity.label in {"ORG", "LOC"}
                    and "person_phrase" in transformations
                    and (entity.label, source_start, source_end) in expansion_evidence
                )
            )
            if supported:
                candidates.append(
                    Candidate(
                        label=entity.label,
                        start=source_start,
                        end=source_end,
                        source="normalized_clean_model",
                        score=entity.score,
                        evidence=("analysis_normalization", *sorted(transformations)),
                    )
                )
        return tuple(candidates)

    def analyze_text(self, text: str) -> AnalysisResult:
        """Normalize user text, predict entities, and preserve zero audio/STT timings."""
        if not isinstance(text, str) or not text.strip():
            raise TextValidationError("Text must be a non-empty string.")
        if len(text) > MAX_TEXT_CHARS:
            raise TextValidationError("Text exceeds the 20,000 character limit.")
        started = self._clock()
        normalization_start = self._clock()
        normalized = self._normalizer(text)
        normalization_ms = self._milliseconds(normalization_start, self._clock())
        ner_start = self._clock()
        entities = validate_entity_spans(normalized, self.ner_predictor.predict(normalized))
        ner_ms = self._milliseconds(ner_start, self._clock())
        return AnalysisResult(
            raw_transcript=text,
            normalized_transcript=normalized,
            entities=entities,
            timing=Timing(
                audio_preprocessing_ms=0.0,
                stt_ms=0.0,
                normalization_ms=normalization_ms,
                ner_ms=ner_ms,
                total_ms=self._milliseconds(started, self._clock()),
            ),
            models=self._models(include_stt=False),
        )

    def analyze_audio(self, audio_path: Path) -> AnalysisResult:
        """Validate/preprocess audio, transcribe it, normalize, and extract entities."""
        if self.stt_service is None:
            raise ModelLoadError("Speech-to-text service is unavailable.")
        started = self._clock()
        preprocessing_start = self._clock()
        with prepared_audio(audio_path, self.audio_config) as canonical_audio:
            preprocessing_ms = self._milliseconds(preprocessing_start, self._clock())
            stt_start = self._clock()
            raw = self.stt_service.transcribe(canonical_audio)
            stt_ms = self._milliseconds(stt_start, self._clock())
        normalization_start = self._clock()
        normalized = self._normalizer(raw)
        normalization_ms = self._milliseconds(normalization_start, self._clock())
        ner_start = self._clock()
        normalized_candidates: tuple[Candidate, ...] = ()
        if self._speech_rescue_enabled and self._analysis_normalization_enabled:
            analysis_view = normalize_speech_analysis(normalized)
            display_predictions, analysis_predictions = self.ner_predictor.predict_many(
                (normalized, analysis_view.analysis_text)
            )
            entities = validate_entity_spans(normalized, display_predictions)
            normalized_candidates = self._normalized_candidates(
                normalized,
                validate_entity_spans(analysis_view.analysis_text, analysis_predictions),
                analysis_view,
                self._normalized_confidence_threshold,
            )
        else:
            entities = validate_entity_spans(normalized, self.ner_predictor.predict(normalized))
        if self._speech_rescue_enabled:
            entities = self._speech_rescue.extract(normalized, entities, normalized_candidates)
        ner_ms = self._milliseconds(ner_start, self._clock())
        return AnalysisResult(
            raw_transcript=raw,
            normalized_transcript=normalized,
            entities=entities,
            timing=Timing(
                audio_preprocessing_ms=preprocessing_ms,
                stt_ms=stt_ms,
                normalization_ms=normalization_ms,
                ner_ms=ner_ms,
                total_ms=self._milliseconds(started, self._clock()),
            ),
            models=self._models(include_stt=True),
        )
