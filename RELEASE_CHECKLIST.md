# Release checklist

This project has two evidence levels. Public fixtures and unit tests are engineering evidence;
they are not a substitute for a consented real-recording benchmark. Do not tag a release or make
end-to-end accuracy claims until every applicable gate below passes.

## Reproducible build

- The intended Python versions are 3.11 and 3.12.
- `pyproject.toml` and `uv.lock` describe the complete normal runtime/development environment.
- Runtime NER artifacts and both Whisper model IDs use their reviewed immutable versions.
- `make format-check`, `make lint`, `make typecheck`, `make coverage`, and
  `make package-check` pass from the release commit.

## Provisioned model verification

- Provision the reviewed local model assets with `make setup-models`.
- Run `make test-slow`; do not replace it with mocked or fixture-only tests.
- Confirm the health response reports the expected STT model ID and immutable revision.

## Private end-to-end evidence

- Every speaker explicitly consented before recording.
- The collection checklist marks each recording, transcript, and entity span independently
  reviewed.
- `make collection-progress` reports `ready_for_evaluation`.
- Run `make release-check` without `--allow-incomplete-dataset`. It must pass all code/model gates,
  validate the private corpus, and regenerate the aggregate STT, pipeline, and error reports.
- Inspect the aggregate reports for the expected sample counts, model revisions, and A-H
  ablations. Keep transcript-bearing rows under `data/private_test/results/`; never commit them.
- If the private corpus is missing or noncompliant, document that limitation exactly as the
  README currently does. Do not infer or fabricate WER, CER, entity F1, or error counts.

## Review and publication

- The pull request is focused, its tests explain the changed behavior, and CI passes on both
  supported Python versions.
- Generated artifacts contain no secrets, absolute private paths, audio, or transcripts.
- Release notes distinguish controlled fixture/NER scores from private real-audio results.
- Prefer new corrective commits for already reviewed public history; do not rewrite shared
  history solely for cosmetic cleanup.
