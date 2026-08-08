from __future__ import annotations

from pathlib import Path

from training.evaluate_ner import resolve_checkpoint


def test_compact_bundle_wins_over_retained_remote_pointer(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.safetensors").write_bytes(b"model")
    (bundle / "labels.json").write_text("{}\n", encoding="utf-8")
    (bundle / "best_checkpoint.json").write_text(
        '{"checkpoint": "checkpoint-816"}\n', encoding="utf-8"
    )

    assert resolve_checkpoint(bundle) == (bundle, bundle)
