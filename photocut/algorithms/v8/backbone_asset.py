"""Strict local provenance and byte verification for the V8 backbone asset."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


class BackboneAssetError(ValueError):
    pass


_FIELDS = {
    "schema_version",
    "asset_id",
    "architecture",
    "weight_enum",
    "source_url",
    "source_repository",
    "source_revision",
    "license",
    "notice_path",
    "notice_sha256",
    "weight_filename",
    "weight_size",
    "weight_sha256",
    "redistributable",
}
_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class BackboneManifest:
    schema_version: int
    asset_id: str
    architecture: str
    weight_enum: str
    source_url: str
    source_repository: str
    source_revision: str
    license: str
    notice_path: str
    notice_sha256: str
    weight_filename: str
    weight_size: int
    weight_sha256: str
    redistributable: bool


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackboneAssetError(f"{name} must be a non-empty string")
    return value.strip()


def _basename(value: Any, name: str) -> str:
    text = _text(value, name)
    if Path(text).name != text:
        raise BackboneAssetError(f"{name} must be a basename")
    return text


def load_backbone_manifest(path: str | Path) -> BackboneManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BackboneAssetError(f"cannot read backbone manifest: {exc}") from exc
    if not isinstance(raw, Mapping) or set(raw) != _FIELDS:
        raise BackboneAssetError("backbone manifest fields differ from schema")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise BackboneAssetError("backbone manifest schema_version must be 1")
    if raw.get("architecture") != "mobilenet_v3_large":
        raise BackboneAssetError("unsupported backbone architecture")
    if type(raw.get("weight_size")) is not int or raw["weight_size"] <= 0:
        raise BackboneAssetError("weight_size must be positive")
    for name in ("weight_sha256", "notice_sha256"):
        if not isinstance(raw.get(name), str) or not _HASH.fullmatch(raw[name]):
            raise BackboneAssetError(f"{name} is invalid")
    if type(raw.get("redistributable")) is not bool:
        raise BackboneAssetError("redistributable must be boolean")
    return BackboneManifest(
        schema_version=1,
        asset_id=_text(raw["asset_id"], "asset_id"),
        architecture="mobilenet_v3_large",
        weight_enum=_text(raw["weight_enum"], "weight_enum"),
        source_url=_text(raw["source_url"], "source_url"),
        source_repository=_text(raw["source_repository"], "source_repository"),
        source_revision=_text(raw["source_revision"], "source_revision"),
        license=_text(raw["license"], "license"),
        notice_path=_basename(raw["notice_path"], "notice_path"),
        notice_sha256=raw["notice_sha256"],
        weight_filename=_basename(raw["weight_filename"], "weight_filename"),
        weight_size=raw["weight_size"],
        weight_sha256=raw["weight_sha256"],
        redistributable=raw["redistributable"],
    )


def _verified_file(root: Path, filename: str, expected_hash: str, expected_size: int | None = None) -> Path:
    candidate = root / filename
    if candidate.is_symlink() or not candidate.is_file():
        raise BackboneAssetError(f"asset file is missing or unsafe: {filename}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise BackboneAssetError(f"asset file escapes manifest root: {filename}")
    before_size = resolved.stat().st_size
    if expected_size is not None and before_size != expected_size:
        raise BackboneAssetError(f"asset size mismatch: {filename}")
    data = resolved.read_bytes()
    if resolved.stat().st_size != before_size:
        raise BackboneAssetError(f"asset changed during verification: {filename}")
    actual = "sha256:" + hashlib.sha256(data).hexdigest()
    if actual != expected_hash:
        raise BackboneAssetError(f"asset hash mismatch: {filename}")
    return resolved


def verify_backbone_asset(manifest: BackboneManifest, manifest_root: str | Path) -> Path:
    root = Path(manifest_root).resolve(strict=True)
    _verified_file(root, manifest.notice_path, manifest.notice_sha256)
    return _verified_file(root, manifest.weight_filename, manifest.weight_sha256, manifest.weight_size)


__all__ = [
    "BackboneAssetError",
    "BackboneManifest",
    "load_backbone_manifest",
    "verify_backbone_asset",
]
