from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from uzbek_speech_entities.ner.predictor import NERPredictor


class _Encoded(dict[str, torch.Tensor]):
    def __init__(self) -> None:
        super().__init__(
            input_ids=torch.ones((3, 4), dtype=torch.long),
            attention_mask=torch.ones((3, 4), dtype=torch.long),
            offset_mapping=torch.tensor(
                [
                    [[0, 0], [0, 1], [0, 0], [0, 0]],
                    [[0, 0], [0, 1], [0, 0], [0, 0]],
                    [[0, 0], [0, 1], [0, 0], [0, 0]],
                ]
            ),
            overflow_to_sample_mapping=torch.tensor([0, 1, 0]),
        )

    def word_ids(self, *, batch_index: int) -> list[int | None]:
        return ([None, 0, None, None], [None, 0, None, None], [None, 1, None, None])[batch_index]


class _Tokenizer:
    def __call__(self, *_: object, **__: object) -> _Encoded:
        return _Encoded()


class _Model:
    config = SimpleNamespace(id2label={0: "O"})

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, **_: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(logits=torch.zeros((3, 4, 1)))


def test_predict_many_batches_overflow_chunks_in_one_forward_and_restores_empty_outputs() -> None:
    predictor = NERPredictor(
        Path("fake"),
        max_length=8,
        confidence_threshold=0.5,
        visible_labels=("PER", "LOC", "ORG", "DATE"),
        model_to_application_labels={"PER": "PER"},
    )
    model = _Model()
    predictor._tokenizer = _Tokenizer()
    predictor._model = model
    predictor._torch = torch
    predictor._device = "cpu"
    predictor._load_attempted = True

    assert predictor.predict_many(("bir ikki", "", "uch")) == ((), (), ())
    assert model.calls == 1
    assert predictor.predict_many(("", "")) == ((), ())
    assert model.calls == 1
