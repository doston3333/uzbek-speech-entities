PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.11)
UV ?= $(if $(wildcard .venv/bin/uv),.venv/bin/uv,uv)

.PHONY: install install-dev setup setup-models check-mps prepare-ner train-ner-clean train-ner-augmented evaluate-ner plan-evaluation import-recordings collection-progress validate-evaluation evaluate-stt evaluate-pipeline error-report test test-slow lint typecheck format format-check coverage package-check release-check run benchmark

RAW_RECORDINGS ?= data/private_test/incoming

install:
	$(UV) sync --locked --no-dev --group tooling --python $(PYTHON)

install-dev:
	$(UV) sync --locked --group dev --group tooling --python $(PYTHON)

# Full local bootstrap on macOS/Linux (venv-aware via PYTHON=).
setup: install
	$(PYTHON) -m uzbek_speech_entities.setup

setup-models:
	$(PYTHON) -m uzbek_speech_entities.setup --models-only

check-mps:
	$(PYTHON) -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')"

prepare-ner:
	$(PYTHON) training/download_ner_dataset.py
	$(PYTHON) training/prepare_ner_dataset.py

train-ner-clean:
	$(PYTHON) training/train_ner.py --config configs/ner_clean.yaml

train-ner-augmented:
	$(PYTHON) training/train_ner.py --config configs/ner_augmented.yaml

evaluate-ner:
	$(PYTHON) training/evaluate_ner.py

plan-evaluation:
	$(PYTHON) -m evaluation.prepare_recording_manifest

import-recordings:
	$(PYTHON) -m evaluation.recording_collection import --raw-directory "$(RAW_RECORDINGS)"

collection-progress:
	$(PYTHON) -m evaluation.recording_collection audit

validate-evaluation:
	$(PYTHON) -m evaluation.dataset --metadata data/private_test/metadata.jsonl --output reports/evaluation_dataset.json

evaluate-stt:
	$(PYTHON) -m evaluation.evaluate_stt --config configs/evaluation.yaml

evaluate-pipeline:
	$(PYTHON) -m evaluation.evaluate_pipeline --config configs/evaluation.yaml

error-report:
	$(PYTHON) -m evaluation.create_error_report --config configs/evaluation.yaml

test:
	$(PYTHON) -W error -m pytest -m "not slow"

test-slow:
	$(PYTHON) -W error -m pytest -m slow

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src training evaluation

format:
	$(PYTHON) -m ruff format .

format-check:
	$(PYTHON) -m ruff format --check .

coverage:
	$(PYTHON) -W error -m pytest -m "not slow" --cov=uzbek_speech_entities

package-check:
	$(PYTHON) scripts/check_wheel.py

# Fail-closed release gate. This intentionally requires provisioned model assets and a
# compliant, consented private evaluation corpus; smoke-only evaluation is never accepted.
release-check:
	$(MAKE) format-check
	$(MAKE) lint
	$(MAKE) typecheck
	$(MAKE) coverage
	$(MAKE) package-check
	$(MAKE) test-slow
	$(MAKE) validate-evaluation
	$(MAKE) evaluate-stt
	$(MAKE) evaluate-pipeline
	$(MAKE) error-report

run:
	$(PYTHON) -m uzbek_speech_entities.api.server

benchmark:
	$(PYTHON) -m evaluation.benchmark_runtime --config configs/evaluation.yaml
