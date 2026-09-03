"""Conservative byte identity for the complete V7/V8 algorithm core."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


_ROOT_CORE_FILES = (
    "core.py",
    "algorithms/v5_2/detector.py",
    "config.py",
)


def _core_paths(project_root: Path) -> tuple[Path, ...]:
    root = project_root.resolve()
    paths = [root / name for name in _ROOT_CORE_FILES]
    for package in ("algorithms/v7", "algorithms/v8"):
        directory = root / package
        if not directory.is_dir():
            raise FileNotFoundError(f"algorithm package is missing: {package}")
        paths.extend(directory.rglob("*.py"))
    asset_directory = root / "algorithms" / "v8" / "assets"
    if asset_directory.is_dir():
        paths.extend(asset_directory.rglob("*.json"))
    unique = tuple(sorted(set(paths), key=lambda path: path.relative_to(root).as_posix()))
    for path in unique:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("core path escapes project root") from exc
        if path.is_symlink():
            raise ValueError(f"core path must not be a symlink: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"algorithm source is missing: {path}")
    return unique


def build_validated_core_manifest(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    files = []
    for path in _core_paths(root):
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        file_sha = "sha256:" + hashlib.sha256(raw).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        digest.update(b"\0")
        files.append({"path": relative, "sha256": file_sha, "size": len(raw)})
    return {
        "schema_version": 1,
        "file_count": len(files),
        "files": files,
        "validated_core_sha256": "sha256:" + digest.hexdigest(),
    }


def compute_validated_core_sha256(project_root: str | Path) -> str:
    return str(build_validated_core_manifest(project_root)["validated_core_sha256"])


__all__ = ["build_validated_core_manifest", "compute_validated_core_sha256"]
