from __future__ import annotations

import pytest

from uzbek_speech_entities.ner.alignment import IGNORE_INDEX, LabelAlignmentError, align_word_labels
from uzbek_speech_entities.ner.labels import build_label_maps
from uzbek_speech_entities.ner.training_data import tokenize_prepared_batch


class FakeEncoding(dict[str, list[list[int]]]):
    def __init__(self, word_ids_by_batch: list[list[int | None]]) -> None:
        super().__init__(input_ids=[[101] * len(word_ids) for word_ids in word_ids_by_batch])
        self._word_ids_by_batch = word_ids_by_batch

    def word_ids(self, batch_index: int) -> list[int | None]:
        return self._word_ids_by_batch[batch_index]


class FakeTokenizer:
    def __init__(self, word_ids_by_batch: list[list[int | None]]) -> None:
        self.word_ids_by_batch = word_ids_by_batch
        self.calls: list[tuple[object, dict[str, object]]] = []

    def __call__(self, tokens: object, **kwargs: object) -> FakeEncoding:
        self.calls.append((tokens, kwargs))
        return FakeEncoding(self.word_ids_by_batch)


def test_alignment_labels_only_first_subtoken_and_ignores_special_tokens() -> None:
    label2id, _ = build_label_maps()

    aligned = align_word_labels(
        [None, 0, 0, 1, 1, 1, 2, None],
        ["B-PER", "I-PER", "O"],
        label2id,
    )

    assert aligned == [
        IGNORE_INDEX,
        label2id["B-PER"],
        IGNORE_INDEX,
        label2id["I-PER"],
        IGNORE_INDEX,
        IGNORE_INDEX,
        label2id["O"],
        IGNORE_INDEX,
    ]


@pytest.mark.parametrize(
    ("word_ids", "tags", "message"),
    [
        ([None, 2, None], ["O", "B-PER"], "does not match"),
        ([None, 0, 2], ["O", "B-PER", "O"], "must not skip"),
        ([None, 0, 1, 0], ["O", "B-PER"], "nondecreasing"),
        ([None, 0], ["B-UNKNOWN"], "unknown or malformed"),
    ],
)
def test_alignment_rejects_invalid_word_ids_and_labels(
    word_ids: list[int | None], tags: list[str], message: str
) -> None:
    label2id, _ = build_label_maps()

    with pytest.raises(LabelAlignmentError, match=message):
        align_word_labels(word_ids, tags, label2id)


def test_batch_tokenization_uses_split_words_and_reports_truncated_entities() -> None:
    label2id, _ = build_label_maps()
    tokenizer = FakeTokenizer([[None, 0, 1, None]])

    result = tokenize_prepared_batch(
        tokenizer,
        {
            "tokens": [["Akmal", "Toshkent", "ertaga"]],
            "ner_tags": [["B-PER", "B-LOC", "B-TEMPORAL"]],
        },
        label2id,
        128,
    )

    assert tokenizer.calls == [
        (
            [["Akmal", "Toshkent", "ertaga"]],
            {"is_split_into_words": True, "truncation": True, "max_length": 128},
        )
    ]
    assert result.model_inputs["labels"] == [
        [IGNORE_INDEX, label2id["B-PER"], label2id["B-LOC"], IGNORE_INDEX]
    ]
    assert result.truncation.as_dict() == {
        "truncated_examples": 1,
        "truncated_words": 1,
        "truncated_entity_words": 1,
    }
