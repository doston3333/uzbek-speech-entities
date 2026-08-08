# Reports

Phase 8 writes aggregate, transcript-free artifacts here:

- `evaluation_collection_plan.json`: planned corpus coverage and explicit not-recorded status;
- `evaluation_collection_progress.json`: local checklist/audio readiness (Git-ignored);
- `evaluation_dataset.json`: corpus compliance evidence;
- `stt_benchmark.csv`: Base/Small WER, CER, mention accuracy, runtime, RTF, and memory;
- `end_to_end_results.csv`: the A–H NER ablation matrix;
- `ner_end_to_end_selection.json`: a recommendation only—never automatic promotion;
- `ner_stability.json`: two-seed clean/augmented checkpoint hashes and stability summaries;
- `error_analysis.md`: taxonomy counts and manual-review guidance.
- `ner_speech_candidate_20260806_v1_release_gate.json`: clean/speech gates and the held-out
  OGG diagnostic for the promoted speech-aware NER checkpoint.

Detailed prediction and error rows contain private transcripts and therefore live under the
Git-ignored `data/private_test/results/` directory. Evaluation commands fail on an incomplete
corpus unless the explicit smoke-only `--allow-incomplete-dataset` flag is used. Smoke outputs
must never be presented as final Phase 8 results.

Pipeline evaluation validates each STT row's configured model ID and requires one non-empty,
consistent resolved revision per model. Audio rows expose surface-mention F1 and a strict aligned
exact-span F1; only exactly preserved STT tokens can be projected into gold coordinates. Error
analysis repeats the dataset compliance gate and labels explicitly allowed incomplete runs as
provisional.
