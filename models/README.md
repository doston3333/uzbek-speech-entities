# Models

Downloaded models, checkpoints, and caches are local-only and ignored by Git.

## Runtime bootstrap (fresh clone)

```bash
# macOS / Linux
./scripts/setup_macos.sh

# Windows PowerShell
.\scripts\setup_windows.ps1
```

This installs Python deps, checks/installs FFmpeg when possible, downloads the pinned NER
inference zip from GitHub Releases (`runtime-models-v1`), and prefetches Whisper Uzbek models
from Hugging Face into `models/cache/`.

You can also run `make setup` or `python -m uzbek_speech_entities.setup` after activating the
venv. Set `SKIP_MODEL_DOWNLOAD=1` to force offline mode.

## Training artifacts

Phase 3 writes the clean run to `models/ner/clean/`. Trainer checkpoints retain model,
optimizer, scheduler, RNG, and Trainer state so `--resume` is a real continuation. The run root
stores the tokenizer, exact package versions, label maps, configuration snapshot, dataset/model
provenance, truncation counts, metrics, duration, and `best_checkpoint.json`. The pointer avoids
duplicating the large best model while keeping the selected checkpoint auditable.
