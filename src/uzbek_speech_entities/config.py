"""Configuration loading that resolves project-relative paths consistently."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _source_checkout_root() -> Path | None:
    """Return the checkout root only when this package is running from one."""
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "configs" / "app.yaml").is_file() else None


def project_root() -> Path:
    """Return the checkout root, or the working directory for an installed wheel."""
    return _source_checkout_root() or Path.cwd().resolve()


def packaged_resource_path(*parts: str) -> Path:
    """Return a package-contained runtime resource path."""
    return Path(__file__).resolve().parent.joinpath(*parts)


def default_config_path() -> Path:
    """Return the checkout default config when present, else its packaged copy."""
    source_root = _source_checkout_root()
    if source_root is not None:
        return source_root / "configs" / "app.yaml"
    return packaged_resource_path("resources", "configs", "app.yaml")


def frontend_directory() -> Path:
    """Return checkout UI assets when present, else their packaged copy."""
    source_root = _source_checkout_root()
    if source_root is not None:
        return source_root / "web"
    return packaged_resource_path("web")


def resolve_project_path(value: str | Path, root: Path | None = None) -> Path:
    """Resolve an absolute path or a path relative to the project root."""
    candidate = Path(value).expanduser()
    base = root or project_root()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class AppConfig:
    """An immutable YAML configuration and the location it was loaded from."""

    path: Path
    values: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a named mapping section or fail with a useful configuration error."""
        section = self.values.get(name)
        if not isinstance(section, Mapping):
            raise ValueError(f"Missing or invalid configuration section: {name}")
        return section


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load a YAML file without importing YAML until configuration is requested."""
    import yaml

    config_path = default_config_path() if path is None else resolve_project_path(path)
    with config_path.open(encoding="utf-8") as config_file:
        values = yaml.safe_load(config_file)
    if not isinstance(values, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return AppConfig(path=config_path, values=_freeze(values))
