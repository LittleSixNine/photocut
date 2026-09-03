"""Resolve PhotoCut's private data directory outside the source checkout."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


DATASET_ENVIRONMENT_VARIABLE = "PHOTOCUT_DATASET_ROOT"
LOCAL_CONFIG_FILENAME = ".photocut-local.json"


def _platform_data_home() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PhotoCut"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        return Path(base) / "PhotoCut" if base else Path.home() / "AppData" / "Local" / "PhotoCut"
    base = os.environ.get("XDG_DATA_HOME")
    return Path(base) / "photocut" if base else Path.home() / ".local" / "share" / "photocut"


def default_dataset_root(project_root: str | Path) -> Path:
    """Return an explicit, local-configured, or platform-standard data path."""
    root = Path(project_root).resolve()
    configured = os.environ.get(DATASET_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()

    config_path = root / LOCAL_CONFIG_FILENAME
    if config_path.is_file() and not config_path.is_symlink():
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid {LOCAL_CONFIG_FILENAME}: {exc}") from exc
        if not isinstance(value, dict) or set(value) != {"dataset_root"}:
            raise ValueError(f"{LOCAL_CONFIG_FILENAME} must contain only dataset_root")
        configured = value["dataset_root"]
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError("dataset_root must be a non-empty path")
        path = Path(configured).expanduser()
        return path if path.is_absolute() else root / path

    return _platform_data_home() / "datasets"


__all__ = [
    "DATASET_ENVIRONMENT_VARIABLE",
    "LOCAL_CONFIG_FILENAME",
    "default_dataset_root",
]
