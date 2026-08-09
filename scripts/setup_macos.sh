#!/usr/bin/env bash
# macOS / Linux bootstrap: venv, Python deps, FFmpeg check, NER + Whisper downloads.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN=python3.11
  elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_BIN=python3.12
  else
    echo "Python 3.11 or 3.12 is required." >&2
    exit 1
  fi
fi

if [[ ! -d .venv ]]; then
  echo "Creating virtualenv with $PYTHON_BIN ..."
  "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .

if ! command -v ffmpeg >/dev/null 2>&1; then
  if [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "Installing FFmpeg via Homebrew..."
    brew install ffmpeg
  else
    echo "FFmpeg not found. On macOS: brew install ffmpeg" >&2
    echo "On Debian/Ubuntu: sudo apt install ffmpeg" >&2
  fi
fi

python -m uzbek_speech_entities.setup
echo
echo "Done. Start the app with: make run"
echo "Open http://127.0.0.1:8000"
