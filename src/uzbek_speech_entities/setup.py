"""Cross-platform runtime setup: system checks plus model downloads."""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import sys
from pathlib import Path

from .config import load_config
from .runtime_models import ensure_runtime_models, ffmpeg_status, ner_bundle_ready
from .stt.base import ModelLoadError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a cloned checkout for local use: verify FFmpeg, download the "
            "pinned NER release, and prefetch Whisper Uzbek models."
        )
    )
    parser.add_argument(
        "--models-only",
        action="store_true",
        help="Skip the FFmpeg advisory and only ensure NER/STT artifacts.",
    )
    parser.add_argument(
        "--force-ner",
        action="store_true",
        help="Re-download the NER release even when models/ner/final already exists.",
    )
    parser.add_argument(
        "--skip-stt",
        action="store_true",
        help="Do not prefetch Whisper models (they can still download on first audio run).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Fail if models are missing instead of downloading.",
    )
    return parser.parse_args(argv)


def _print_platform_banner() -> None:
    print(f"Platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:   {sys.version.split()[0]} ({sys.executable})")


def _print_ffmpeg_advice() -> int:
    ok, detail = ffmpeg_status()
    if ok:
        print(f"FFmpeg:   OK ({detail})")
        return 0
    print(f"FFmpeg:   missing — {detail}")
    print(
        "FFmpeg is optional for WAV/FLAC via libsndfile, but required for many "
        "M4A/WebM/OGG uploads."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _print_platform_banner()
    if not args.models_only:
        _print_ffmpeg_advice()

    try:
        config = load_config()
        result = ensure_runtime_models(
            config,
            local_files_only=args.local_files_only,
            force_ner=args.force_ner,
            prefetch_stt=not args.skip_stt,
        )
    except (ModelLoadError, ValueError, OSError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1

    print(f"NER:      {result['ner_path']}")
    if not ner_bundle_ready(Path(result["ner_path"])):
        print("Setup failed: NER bundle incomplete after install.", file=sys.stderr)
        return 1
    if result["stt_models"]:
        print("STT:      " + ", ".join(result["stt_models"]))
    elif args.skip_stt:
        print("STT:      skipped (--skip-stt)")
    else:
        print("STT:      no models prefetched")
    if shutil.which("ffmpeg"):
        print("Ready:    run `make run` (macOS/Linux) or `.\\scripts\\run_windows.ps1` (Windows).")
    else:
        print(
            "Ready:    models are installed; install FFmpeg for broader audio formats, "
            "then start the app."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
