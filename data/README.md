# Data

Keep source audio and private evaluation recordings local. Do not commit recordings or
transcripts without explicit authorization.

## Public NER training data

Phase 2 uses the public [uznlp-uz/uzbek_NER](https://huggingface.co/datasets/uznlp-uz/uzbek_NER)
dataset, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the
dataset to Elov B.B. and Alaev R.H. (2026), the authors named by the dataset card, whenever
using or redistributing derived data.

The reproducible source is pinned to revision
`4825ce29a1372bd78cb1cbb73693f16ec6f8328d`:

```text
https://huggingface.co/datasets/uznlp-uz/uzbek_NER/resolve/4825ce29a1372bd78cb1cbb73693f16ec6f8328d/Uzbek_NER_Gold.tsv?download=true
```

Its expected SHA-256 is
`45acfdc7fabb8668b383b1fee31d2541c767aa8d05e9a0b1e70475f1b424eac8`.

The TSV schema is exactly `Sentence` (positive integer), `TokenOrder` (positive, 1-based
integer), `Token` (string), `NER_Tag` (BIO label), and `pos` (string). The preparer checks this
schema, reconstructs each sentence by numeric `TokenOrder`, and rejects malformed rows or full
sentences rather than repairing them.

Run the reproducible pipeline from the repository root:

```bash
make prepare-ner
```

Or run the stages independently:

```bash
python training/download_ner_dataset.py
python training/prepare_ner_dataset.py
```

The verified source TSV is local-only at `data/raw/Uzbek_NER_Gold.tsv`. Preparation writes the
deterministic sentence-level (seed `42`) 80/10/10 split to `data/processed/ner/`:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `statistics.json`
- `rejected.jsonl`

The preparer verifies the pinned checksum again and records it in `statistics.json`. Rejected
diagnostics retain the malformed source row or prepared record so annotation problems remain
auditable without silently repairing them.

## Public V5 continuation sources

The optional Modal V5 continuation adds five independently pinned public sources without
changing the validation or test splits above:

- `UzNER-Style(v2)` from Mendeley Data, DOI `10.17632/48923w3gyr.1`, CC BY 4.0. The workbook URL
  is pinned in `public_corpora.py` and must match SHA-256
  `d0d50fa1dfb83cd66abf39076207968681e67ee80814301fa3248f256ff171d0`.
- `yakhyo/mozilla-common-voice-uzbek`, train transcript column only, revision
  `09f89fbf98a7d73a394ae80921950966a5569c1c`, with the dataset card's inherited CC0 claim.
- `murodbek/uzbek-speech-corpus`, train transcript column only, revision
  `257f7e46f0a92d81ba00f22659ec93213a3b5f7e`, CC BY 4.0.
- `islomov/news_youtube_uzbek_speech_dataset`, train transcript column only, revision
  `bbff3fb27cbf461260f2b5f93e5f56d0c4008a6c`, Apache-2.0.
- `ai4uz/uzbekvoice-filtered`, train transcript column only, revision
  `b392eae07f28911b1538215c130bf056f7b2f7fa`, Apache-2.0.

The speech adapters use HTTP range reads for only the allowlisted text column; they never request
or persist public audio. They accept complete, explicit number-word calendar years followed by an
Uzbek `yil` suffix. Because ASR corpora commonly normalize the same spoken content to digits, an
audited transformation also converts only `1800`--`2035` digit years immediately followed by a
year suffix into canonical Uzbek spoken words; duration contexts remain excluded. The finalizer
keeps the promoted speech-aware training file as the immutable base, caps the broader public NER
slices, removes normalized exact overlap with protected evaluation text, and holds back a
deterministic per-source speech-year dev partition.

For `N` accepted sentences, the split uses `floor(0.8*N)` train,
`floor(0.1*N)` validation, and assigns the remainder to test. Raw and processed artifacts are
ignored by Git.
