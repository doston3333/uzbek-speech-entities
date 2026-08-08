# Private evaluation data

Place consented local evaluation recordings in `audio/`; recordings, the real metadata, the
recording checklist, and detailed evaluation outputs are ignored by Git. Entity offsets use
Python's zero-based, end-exclusive character indexing into `gold_transcript`.

The committed `prompts.jsonl` and `speakers.example.json` define a 30-prompt × 5-speaker plan.
Replace the placeholder speaker slots only with people who have explicitly agreed to record,
then generate the local manifest and checklist:

```bash
make plan-evaluation
```

This command creates `metadata.jsonl`, `recording_checklist.csv`, and
`reports/evaluation_collection_plan.json`. It refuses to overwrite them unless the module is run
with `--overwrite`. A generated plan is not evaluation evidence: its status is
`planned_not_recorded`, its recorded count is zero, and it creates no audio. For every row,
confirm consent first, record the named WAV file, then independently review the transcript and
entity spans before checking the four checklist fields.

## Import consented recordings

Each recording must be 5–20 seconds and use its checklist ID as the filename, for example
`speaker-01-prompt-01.m4a`. WAV, MP3, M4A, WebM, and FLAC sources are accepted. Before importing
any file, manually change `consent_confirmed` to `True` on that exact checklist row—but only after
that speaker has explicitly agreed to participate. Put source files in the ignored `incoming/`
directory or pass another directory:

```bash
make import-recordings
# or
make import-recordings RAW_RECORDINGS=/absolute/path/to/recordings
```

The importer never deletes a source recording. It rejects unknown or duplicate IDs, missing
consent, unreadable or oversized files, and durations outside 5–20 seconds. Accepted recordings
are converted to uniform 16 kHz mono PCM WAV files beneath `audio/`, and the corresponding
`recorded` fields become `True`. Existing valid audio is kept unchanged. Replacement requires a
separate, explicit `--overwrite` invocation of `evaluation.recording_collection`.

After listening to each saved recording, correct the gold transcript if the speaker departed from
the prompt, re-check every entity offset, and only then set `transcript_reviewed` and
`entity_spans_reviewed` to `True`. Check aggregate progress without emitting transcripts:

```bash
make collection-progress
```

Do not run the final model comparison until this reports `ready_for_evaluation` with 150/150 rows.

The validator refuses malformed paths, duplicate IDs/files, invalid or overlapping spans, text
mismatches, missing audio, and unreadable audio. It then reports—without inventing coverage—
whether the corpus has:

- 100–300 recordings of 5–20 seconds;
- at least five speakers where practical;
- at least 30 PER, LOC, ORG, and DATE mentions;
- every acoustic/speech condition and content category required by the build specification.

Conditions use stable snake-case tags. Acoustic/speech tags are `quiet`, `background_noise`,
`laptop_microphone`, `phone_microphone`, `fast_speech`, `slow_speech`, `formal_speech`,
`conversational_speech`, and `uzbek_russian_code_switching`. Content tags are `common_names`,
`rare_names`, `cities`, `regions`, `districts`, `villages`, `organizations`, `relative_dates`,
`numeric_dates`, `times`, and `inflected_entities`.

Validate before any model run:

```bash
make validate-evaluation
```

The committed reports contain aggregate metrics only. Complete transcripts and per-recording
predictions are written beneath `data/private_test/results/`, remain local, and are used only
for the explicit evaluation/error-analysis workflow.
