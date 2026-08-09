"""Deterministic model-free services for Phase 6 tests."""

from __future__ import annotations

from pathlib import Path

from uzbek_speech_entities.ner.schemas import Entity


class FakeSTTService:
    def __init__(self, transcript: str = "Akmal Toshkentga bordi.") -> None:
        self._transcript = transcript
        self._loaded = True
        self.transcribed_paths: list[Path] = []

    @property
    def model_id(self) -> str:
        return "fake-stt"

    @property
    def revision(self) -> str:
        return "0000000000000000000000000000000000000000"

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str | None:
        return "cpu"

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, audio_path: Path) -> str:
        self.transcribed_paths.append(audio_path)
        return self._transcript


class FakeNERPredictor:
    def __init__(self, entities: tuple[Entity, ...] = ()) -> None:
        self.entities = entities
        self._loaded = True
        self.inputs: list[str] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str | None:
        return "cpu"

    @property
    def model_path(self) -> Path:
        return Path("models/ner/fake")

    def load(self) -> None:
        self._loaded = True

    def predict(self, text: str) -> tuple[Entity, ...]:
        self.inputs.append(text)
        return self.entities

    def predict_many(self, texts: tuple[str, ...]) -> tuple[tuple[Entity, ...], ...]:
        self.inputs.extend(texts)
        return tuple(self.entities for _ in texts)
