## Problem and outcome

Describe the concrete problem, its root cause, and the resulting behavior.

## Scope

- [ ] The change is focused; unrelated cleanup is excluded or clearly separated.
- [ ] Config, model, API, and packaging compatibility impacts are documented.
- [ ] No private audio/transcripts, credentials, or unreviewed model artifacts are included.

## Verification

- [ ] `make format-check`
- [ ] `make lint`
- [ ] `make typecheck`
- [ ] `make coverage`
- [ ] `make package-check`
- [ ] Tests were added or updated for behavior changes.
- [ ] `make test-slow` was run when model-backed behavior changed, or the reason it was not run is
      stated below.

## Evidence and claims

- [ ] Model and dataset revisions are immutable and recorded where applicable.
- [ ] Provisional fixture/model scores are not presented as private end-to-end results.
- [ ] Release-impacting changes satisfy `RELEASE_CHECKLIST.md`, including consented private data
      gates when making real-audio claims.

## Notes and residual risk

List any platform, model, data, or deployment checks that could not be run.
