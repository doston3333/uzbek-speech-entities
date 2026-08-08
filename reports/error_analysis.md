# Error analysis

No private evaluation has been run. Populate `data/private_test/metadata.jsonl` with a compliant,
consented corpus, run `make evaluate-stt`, `make evaluate-pipeline`, and then `make error-report`.

The generated aggregate report contains no transcripts. Complete per-recording records remain in
the Git-ignored `data/private_test/results/error_analysis.jsonl` file.
