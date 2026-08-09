"""Local Transformers token-classification predictor for normalized Uzbek text."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from ..config import AppConfig, resolve_project_path
from ..stt.base import ModelLoadError
from .schemas import Entity
from .spans import NERPredictionError, TokenPrediction, aggregate_bio_predictions

LOGGER = logging.getLogger(__name__)
_LEXICAL_UNITS = re.compile(r"\w+(?:[ʻ’‘ʼ'`:/.-]\w+)*|[^\w\s]", re.UNICODE)


@runtime_checkable
class NERService(Protocol):
    """Minimal interface consumed by the framework-independent analyzer."""

    @property
    def loaded(self) -> bool: ...

    @property
    def device(self) -> str | None: ...

    @property
    def model_path(self) -> Path: ...

    def load(self) -> None: ...

    def predict(self, text: str) -> tuple[Entity, ...]: ...

    def predict_many(self, texts: Sequence[str]) -> tuple[tuple[Entity, ...], ...]: ...


class NERPredictor:
    """One-shot local NER model loader with word-first-subtoken decoding."""

    def __init__(
        self,
        model_path: Path,
        *,
        max_length: int,
        confidence_threshold: float,
        visible_labels: Sequence[str],
        model_to_application_labels: Mapping[str, str],
        local_files_only: bool = False,
    ) -> None:
        if max_length < 8 or not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("NER max_length or confidence threshold is invalid.")
        self._model_path = Path(model_path)
        self.max_length = max_length
        self.confidence_threshold = float(confidence_threshold)
        self.visible_labels = frozenset(visible_labels)
        self.model_to_application_labels = dict(model_to_application_labels)
        if not self.visible_labels <= {"PER", "LOC", "ORG", "DATE"}:
            raise ValueError("NER visible labels must be public labels.")
        self.local_files_only = local_files_only
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: str | None = None
        self._load_error: ModelLoadError | None = None
        self._load_attempted = False
        self._lock = RLock()

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    @classmethod
    def from_config(cls, config: AppConfig, *, local_files_only: bool = False) -> NERPredictor:
        values = config.section("ner")
        configured_path = values.get("model_path")
        override = os.getenv("NER_MODEL_PATH")
        path_value = override.strip() if override else configured_path
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError("NER model path must be a non-empty string.")
        max_length = values.get("max_length")
        threshold = values.get("confidence_threshold")
        labels = values.get("visible_labels")
        mapping = values.get("model_to_application_labels")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or isinstance(threshold, bool)
            or not isinstance(threshold, int | float)
            or isinstance(labels, str | bytes)
            or not isinstance(labels, Sequence)
            or not isinstance(mapping, Mapping)
        ):
            raise ValueError("NER configuration is invalid.")
        if not all(isinstance(item, str) for item in labels) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
        ):
            raise ValueError("NER label configuration is invalid.")
        model_path = resolve_project_path(path_value)
        if not local_files_only:
            from ..runtime_models import ensure_ner_model

            model_path = ensure_ner_model(model_path, config, local_files_only=local_files_only)
        return cls(
            model_path,
            max_length=max_length,
            confidence_threshold=float(threshold),
            visible_labels=tuple(labels),
            model_to_application_labels={str(key): str(value) for key, value in mapping.items()},
            local_files_only=local_files_only,
        )

    def _dependencies(self) -> tuple[Any, Any, Any]:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        return torch, AutoTokenizer, AutoModelForTokenClassification

    @staticmethod
    def _select_device(torch_module: Any) -> str:
        if bool(torch_module.backends.mps.is_available()):
            LOGGER.info("Using MPS for NER inference.")
            return "mps"
        LOGGER.warning("Using CPU for NER inference because MPS is unavailable.")
        return "cpu"

    def load(self) -> None:
        """Load tokenizer and model once; repeated calls reuse the outcome."""
        if self._load_attempted:
            if self._load_error is not None:
                raise self._load_error
            return
        with self._lock:
            if self._load_attempted:
                if self._load_error is not None:
                    raise self._load_error
                return
            self._load_attempted = True
            try:
                torch_module, tokenizer_class, model_class = self._dependencies()
                device = self._select_device(torch_module)
                tokenizer = tokenizer_class.from_pretrained(
                    self.model_path,
                    use_fast=True,
                    local_files_only=self.local_files_only,
                    # This artifact uses BertPreTokenizer. Transformers otherwise
                    # emits a Mistral-regex warning whose suggested patch is
                    # incompatible with (and raises for) the trained tokenizer.
                    fix_mistral_regex=False,
                )
                if not getattr(tokenizer, "is_fast", False):
                    raise ModelLoadError("NER requires a fast tokenizer.")
                model = model_class.from_pretrained(
                    self.model_path, local_files_only=self.local_files_only
                )
                model.to(device)
                model.eval()
                self._torch, self._tokenizer, self._model, self._device = (
                    torch_module,
                    tokenizer,
                    model,
                    device,
                )
                LOGGER.info("Loaded NER model from %s on %s.", self.model_path, device)
            except ModelLoadError as error:
                self._load_error = error
                raise
            except Exception as error:
                self._load_error = ModelLoadError("NER model could not be loaded.")
                raise self._load_error from error

    @staticmethod
    def lexical_units(text: str) -> tuple[tuple[str, int, int], ...]:
        """Return punctuation-aware lexical units and absolute character offsets."""
        return tuple(
            (match.group(), match.start(), match.end()) for match in _LEXICAL_UNITS.finditer(text)
        )

    def predict(self, text: str) -> tuple[Entity, ...]:
        """Predict one input through the compatibility wrapper."""
        return self.predict_many((text,))[0]

    def predict_many(self, texts: Sequence[str]) -> tuple[tuple[Entity, ...], ...]:
        """Predict multiple inputs with one forward pass across overflow chunks."""
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
            raise NERPredictionError("NER inputs must be a sequence of text.")
        if not all(isinstance(text, str) for text in texts):
            raise NERPredictionError("NER input must be text.")
        units_by_sample = tuple(self.lexical_units(text) for text in texts)
        nonempty = tuple(index for index, units in enumerate(units_by_sample) if units)
        if not nonempty:
            return tuple(() for _ in texts)
        self.load()
        if (
            self._tokenizer is None
            or self._model is None
            or self._torch is None
            or self._device is None
        ):
            raise NERPredictionError("NER model is unavailable.")
        words = [[word for word, _, _ in units_by_sample[index]] for index in nonempty]
        stride = min(max(1, self.max_length // 4), self.max_length - 2)
        try:
            encoded = self._tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=self.max_length,
                stride=stride,
                padding=True,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mappings = encoded.pop("offset_mapping")
            overflow_to_sample_mapping = encoded.pop("overflow_to_sample_mapping")
            id2label = self._model.config.id2label
            chosen: dict[tuple[int, int], tuple[float, str]] = {}
            chunk_count = int(encoded["input_ids"].shape[0])
            model_inputs = {
                key: value.to(self._device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            with self._torch.inference_mode():
                logits = self._model(**model_inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)
            for batch_index in range(chunk_count):
                sample_index = nonempty[int(overflow_to_sample_mapping[batch_index].item())]
                word_ids = encoded.word_ids(batch_index=batch_index)
                prior_word: int | None = None
                for token_index, word_id in enumerate(word_ids):
                    if word_id is None or word_id == prior_word:
                        prior_word = word_id
                        continue
                    prior_word = word_id
                    relative_start = int(offset_mappings[batch_index, token_index, 0].item())
                    if relative_start != 0:
                        continue
                    token_probabilities = probabilities[batch_index, token_index]
                    label_id = int(token_probabilities.argmax().item())
                    label = str(id2label[label_id])
                    probability = float(token_probabilities[label_id].item())
                    key = (sample_index, word_id)
                    prior = chosen.get(key)
                    if prior is None or probability > prior[0]:
                        chosen[key] = (probability, label)
            results: list[tuple[Entity, ...]] = [() for _ in texts]
            for sample_index in nonempty:
                units = units_by_sample[sample_index]
                sample_keys = {word_id for index, word_id in chosen if index == sample_index}
                if sample_keys != set(range(len(units))):
                    raise NERPredictionError("NER tokenization omitted part of the input text.")
                predictions = [
                    TokenPrediction(
                        label=label, score=score, start=units[word_id][1], end=units[word_id][2]
                    )
                    for (_, word_id), (score, label) in sorted(
                        chosen.items(), key=lambda item: item[0]
                    )
                    if _ == sample_index
                ]
                results[sample_index] = aggregate_bio_predictions(
                    texts[sample_index],
                    predictions,
                    model_to_application_labels=self.model_to_application_labels,
                    visible_labels=self.visible_labels,
                    threshold=self.confidence_threshold,
                )
            return tuple(results)
        except NERPredictionError:
            raise
        except Exception as error:
            raise NERPredictionError("NER prediction could not be completed.") from error
