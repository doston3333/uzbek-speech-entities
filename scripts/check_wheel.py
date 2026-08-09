"""Build and verify the distributable wheel without loading or downloading models."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _venv_python(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="uzbek-wheel-check-") as temp_name:
        temp_root = Path(temp_name)
        wheel_directory = temp_root / "wheel"
        subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_directory)],
            check=True,
            cwd=root,
        )
        wheels = tuple(wheel_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("wheel build did not produce exactly one wheel")

        environment = temp_root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = _venv_python(environment)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
            check=True,
        )
        unrelated_cwd = temp_root / "unrelated"
        unrelated_cwd.mkdir()
        verification = """
from importlib.resources import files
import uzbek_speech_entities
from uzbek_speech_entities.config import default_config_path, frontend_directory

config_path = default_config_path()
assert config_path.is_file()
assert "stt:" in config_path.read_text(encoding="utf-8")
web_directory = frontend_directory()
for name in ("index.html", "styles.css", "app.js"):
    assert (web_directory / name).is_file(), name
package = files("uzbek_speech_entities")
for name in (
    "resources/configs/app.yaml",
    "web/index.html",
    "web/styles.css",
    "web/app.js",
    "ner/resources/temporal_terms.json",
):
    assert package.joinpath(name).is_file(), name
"""
        subprocess.run([str(python), "-c", verification], check=True, cwd=unrelated_cwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
