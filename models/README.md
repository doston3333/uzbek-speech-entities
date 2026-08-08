# Models

Downloaded models, checkpoints, and caches are local-only and ignored by Git.

Phase 3 writes the clean run to `models/ner/clean/`. Trainer checkpoints retain model,
optimizer, scheduler, RNG, and Trainer state so `--resume` is a real continuation. The run root
stores the tokenizer, exact package versions, label maps, configuration snapshot, dataset/model
provenance, truncation counts, metrics, duration, and `best_checkpoint.json`. The pointer avoids
duplicating the large best model while keeping the selected checkpoint auditable.
