PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.11)

.PHONY: install install-dev setup setup-models check-mps prepare-ner train-ner-clean train-ner-augmented evaluate-ner plan-evaluation import-recordings collection-progress validate-evaluation evaluate-stt evaluate-pipeline error-report test test-slow lint typecheck run benchmark

RAW_RECORDINGS ?= data/private_test/incoming

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install --no-deps -e .

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
	$(PYTHON) -m pytest -m "not slow"

test-slow:
	$(PYTHON) -m pytest -m slow

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src training evaluation

run:
	$(PYTHON) -m uzbek_speech_entities.api.server

benchmark:
	$(PYTHON) -m evaluation.benchmark_runtime --config configs/evaluation.yaml
