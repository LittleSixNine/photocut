"""Strict manifests for optional, explicitly supplied V7 model artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


class ModelManifestError(ValueError):
    """The manifest is malformed or outside the supported contract."""


class ModelArtifactError(ValueError):
    """The model file does not match its verified manifest."""


_FIELDS = {
    "schema_version", "model_id", "adapter", "source_repository",
    "source_revision", "license", "notice_path", "model_filename",
    "model_size", "model_sha256", "input_name", "input_size",
    "output_names", "redistributable",
}
_ADAPTERS = {"docquadnet", "docaligner_heatmap", "photo_mask_v8"}
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_id: str
    adapter: str
    source_repository: str
    source_revision: str
    license: str
    notice_path: str
    model_filename: str
    model_size: int
    model_sha256: str
    input_name: str
    input_size: tuple[int, int]
    output_names: tuple[str, ...]
    redistributable: bool


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"{name} must be a non-empty string")
    return value


def load_model_manifest(path: str | Path) -> ModelManifest:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"cannot read model manifest: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ModelManifestError("model manifest must be an object")
    keys = set(raw)
    if keys != _FIELDS:
        missing = sorted(_FIELDS - keys)
        unknown = sorted(keys - _FIELDS)
        raise ModelManifestError(f"manifest fields differ; missing={missing}, unknown={unknown}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ModelManifestError("schema_version must be integer 1")
    adapter = _text(raw["adapter"], "adapter")
    if adapter not in _ADAPTERS:
        raise ModelManifestError("unsupported model adapter")
    model_size = raw["model_size"]
    if type(model_size) is not int or model_size <= 0:
        raise ModelManifestError("model_size must be a positive integer")
    model_sha256 = _text(raw["model_sha256"], "model_sha256")
    if not _SHA256.fullmatch(model_sha256):
        raise ModelManifestError("model_sha256 must be sha256:<64 lowercase hex>")
    input_size = raw["input_size"]
    if (not isinstance(input_size, list) or len(input_size) != 2 or
            any(type(item) is not int or item <= 0 for item in input_size)):
        raise ModelManifestError("input_size must contain two positive integers")
    output_names = raw["output_names"]
    if (not isinstance(output_names, list) or not output_names or
            any(not isinstance(item, str) or not item.strip() for item in output_names) or
            len(set(output_names)) != len(output_names)):
        raise ModelManifestError("output_names must contain unique non-empty strings")
    redistributable = raw["redistributable"]
    if type(redistributable) is not bool:
        raise ModelManifestError("redistributable must be boolean")
    model_filename = _text(raw["model_filename"], "model_filename")
    if Path(model_filename).name != model_filename:
        raise ModelManifestError("model_filename must be a basename")
    return ModelManifest(
        schema_version=1,
        model_id=_text(raw["model_id"], "model_id"),
        adapter=adapter,
        source_repository=_text(raw["source_repository"], "source_repository"),
        source_revision=_text(raw["source_revision"], "source_revision"),
        license=_text(raw["license"], "license"),
        notice_path=_text(raw["notice_path"], "notice_path"),
        model_filename=model_filename,
        model_size=model_size,
        model_sha256=model_sha256,
        input_name=_text(raw["input_name"], "input_name"),
        input_size=(input_size[0], input_size[1]),
        output_names=tuple(output_names),
        redistributable=redistributable,
    )


def verify_model_artifact(manifest: ModelManifest, path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_symlink():
        raise ModelArtifactError("model artifact must not be a symlink")
    if not candidate.exists():
        raise ModelArtifactError("model artifact is missing")
    if not candidate.is_file():
        raise ModelArtifactError("model artifact is not a regular file")
    if candidate.name != manifest.model_filename:
        raise ModelArtifactError("model artifact filename does not match manifest")
    if candidate.stat().st_size != manifest.model_size:
        raise ModelArtifactError("model artifact size mismatch")
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelArtifactError(f"cannot read model artifact: {exc}") from exc
    expected = manifest.model_sha256.removeprefix("sha256:")
    if digest.hexdigest() != expected:
        raise ModelArtifactError("model artifact hash mismatch")
    return candidate.resolve(strict=True)
