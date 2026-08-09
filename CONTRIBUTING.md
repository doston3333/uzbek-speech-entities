# Contributing

Thanks for helping improve Uzbek Speech Entity Extractor.

## Development setup

Prerequisites: Python 3.11 or 3.12, and FFmpeg when libsndfile cannot decode
M4A/WebM/OGG directly (`brew install ffmpeg` on macOS).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install "uv==0.9.28"
make install-dev
make setup-models   # NER GitHub Release + Whisper Uzbek prefetch
```

`uv.lock` is the reproducible dependency source for normal runtime, development, and the pinned
`uv` tooling bootstrap. `requirements-modal.txt` is reserved for the isolated Modal image
workflow.

On macOS/Linux you can instead run `./scripts/setup_macos.sh`. On Windows use
`.\scripts\setup_windows.ps1`. Pass `PYTHON=python3.12` to Make targets if you prefer
Python 3.12.

Copy `.env.example` if you need local overrides; never commit real secrets.

## Checks before opening a PR

```bash
python -m compileall src training evaluation
make test
make lint
make typecheck
make format-check
make coverage
make package-check
```

- `make test` runs the default pytest selection (`-m "not slow"`) and does not
  download or load Whisper/NER checkpoints; warnings are errors.
- `make test-slow` needs local model artifacts and stays local-files-only.
- `make lint` runs Ruff; `make typecheck` runs mypy on `src`, `training`, and
  `evaluation`.
- `make format-check` verifies Ruff formatting; `make coverage` enforces 80% branch coverage;
  `make package-check` verifies an isolated built wheel and packaged resources.

CI runs the locked lightweight path on Python 3.11 and 3.12. The separate
`Provisioned model tests` workflow downloads/caches the pinned model assets and runs the slow
suite on explicit dispatch. Private evaluation data is never uploaded to CI.

## Pull request expectations

- Keep changes focused; prefer small PRs with a clear problem statement.
- Match existing module layout, naming, and config-driven behavior.
- Add or update unit tests for behavior changes when practical.
- Keep each commit to one reviewable concern. Put mechanical formatting or generated lock-file
  changes in the same commit only when they are required by that concern.
- Write commit messages that state the behavior or invariant changed, and avoid force-pushing
  reviewed public history merely to make it look cleaner.
- Do not commit private audio, transcripts, checkpoints under `models/`, or
  credentials.
- Do not treat provisional NER fixture scores as end-to-end product claims.
- Complete every applicable item in the pull-request template. A release additionally requires
  every gate in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Report security issues privately via [SECURITY.md](SECURITY.md), not public
  issues.

## Running the app locally

```bash
make setup-models   # once per machine / when models/ is empty
make run
```

On Windows: `.\scripts\run_windows.ps1`. Open <http://127.0.0.1:8000>. Host and port
come from `configs/app.yaml` (`APP_HOST` / `APP_PORT` override).
