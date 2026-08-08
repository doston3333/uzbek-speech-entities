from __future__ import annotations

import pytest

from uzbek_speech_entities.ner.offset_tokens import comparison_key, tokenize_words


def test_word_tokens_keep_exact_displayed_offsets_and_fold_apostrophes() -> None:
    text = "Yangi oʻzbekiston, O'zbekiston!"
    tokens = tokenize_words(text)
    assert [(token.text, token.start, token.end) for token in tokens] == [
        ("Yangi", 0, 5),
        ("oʻzbekiston", 6, 17),
        ("O'zbekiston", 19, 30),
    ]
    assert comparison_key("oʻzbekiston") == comparison_key("O'zbekiston")
    assert comparison_key("Ｏ‘ZBEKISTON") == comparison_key("oʻzbekiston")
    with pytest.raises(TypeError):
        tokenize_words(42)  # type: ignore[arg-type]
