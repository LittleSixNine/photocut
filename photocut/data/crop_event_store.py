"""Append-only, idempotent provenance records for production crop attempts."""

import copy
import hashlib
import json
import math
import os
import stat
from datetime import datetime
from numbers import Real
from pathlib import Path

from photocut.confirmation.model import _non_empty_id, _validate_integer_corners
from photocut.data.dataset_store import append_jsonl_validated, load_jsonl


_REQUIRED_FIELDS = {
    "schema_version", "event_id", "crop_id", "image_id", "annotation_id",
    "boundary_corners", "crop_corners", "inset", "output_path", "status",
    "error", "created_at",
}
_IDEMPOTENCY_FIELDS = (
    "image_id", "annotation_id", "boundary_corners", "crop_corners", "inset",
    "status",
)


def _finite_pair(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly four 2D points")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain exactly four 2D points")
    return [float(values[0]), float(values[1])]


def _validate_numeric_corners(values, name):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{name} must contain exactly four 2D points")
    return [_finite_pair(point, name) for point in values]


def _validate_timestamp(value):
    if not isinstance(value, str) or not value:
        raise ValueError("created_at must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("created_at must be a timezone-aware ISO timestamp") from None
    if parsed.utcoffset() is None:
        raise ValueError("created_at must be a timezone-aware ISO timestamp")
    return value


def crop_event_key(event):
    payload = {name: event[name] for name in _IDEMPOTENCY_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_event_id(event):
    return f"crop_{crop_event_key(event)}"


def validate_crop_event(event, *, require_deterministic_ids=False):
    if not isinstance(event, dict) or set(event) != _REQUIRED_FIELDS:
        raise ValueError("crop event schema has missing or unexpected fields")
    if type(event["schema_version"]) is not int or event["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    boundary = _validate_integer_corners(event["boundary_corners"], "boundary_corners")
    crop = _validate_numeric_corners(event["crop_corners"], "crop_corners")
    inset = event["inset"]
    if not isinstance(inset, dict) or set(inset) != {"method", "distance_px"} or inset["method"] != "parallel_edge_offset":
        raise ValueError("invalid crop inset method")
    distance = inset["distance_px"]
    if isinstance(distance, bool) or not isinstance(distance, Real) or not math.isfinite(distance) or distance < 0:
        raise ValueError("invalid crop inset distance")
    status = event["status"]
    if status not in {"succeeded", "failed"}:
        raise ValueError("invalid crop event status")
    output_path = event["output_path"]
    error = event["error"]
    if status == "succeeded":
        if not isinstance(output_path, str) or not output_path:
            raise ValueError("successful crop event requires output_path")
        if error is not None:
            raise ValueError("successful crop event cannot contain error")
    elif output_path is not None or not isinstance(error, str) or not error.strip():
        raise ValueError("failed crop event requires error and no output_path")
    event_id = _non_empty_id(event["event_id"], "event_id")
    crop_id = _non_empty_id(event["crop_id"], "crop_id")
    normalized = {
        "schema_version": 1,
        "event_id": "",
        "crop_id": "",
        "image_id": _non_empty_id(event["image_id"], "image_id"),
        "annotation_id": _non_empty_id(event["annotation_id"], "annotation_id"),
        "boundary_corners": boundary,
        "crop_corners": crop,
        "inset": {"method": "parallel_edge_offset", "distance_px": float(distance)},
        "output_path": output_path,
        "status": status,
        "error": error,
        "created_at": _validate_timestamp(event["created_at"]),
    }
    expected_id = deterministic_event_id(normalized)
    if require_deterministic_ids and (event_id != expected_id or crop_id != expected_id):
        raise ValueError("crop event IDs must match the deterministic event ID")
    normalized["event_id"] = expected_id
    normalized["crop_id"] = expected_id
    return normalized


class _ExistingCropEvent(Exception):
    def __init__(self, event):
        self.event = event


class CropEventStore:
    def __init__(self, path: Path, *, batch_dir: Path | None = None):
        raw_path = Path(path)
        if ".." in raw_path.parts:
            raise ValueError("crop event path must not contain parent traversal")
        self.path = Path(os.path.abspath(raw_path))
        self.batch_dir = None if batch_dir is None else Path(os.path.abspath(batch_dir))
        if self.batch_dir is not None and self.path != self.batch_dir / "crops.jsonl":
            raise ValueError("crop event path must be the batch crops.jsonl file")

    def _validate_path(self):
        directory = self.path.parent
        if self.batch_dir is not None and directory != self.batch_dir:
            raise ValueError("crop event path escapes its batch directory")
        try:
            info = directory.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("crop event path must have a regular parent directory")

    def append_idempotent(self, event):
        self._validate_path()
        normalized = validate_crop_event(event)

        def validate_existing(existing):
            normalized_existing = [
                validate_crop_event(item, require_deterministic_ids=True)
                for item in existing
            ]
            if any(item["event_id"] == normalized["event_id"] for item in normalized_existing):
                for item in normalized_existing:
                    if item["event_id"] == normalized["event_id"]:
                        raise _ExistingCropEvent(item)
            if any(item["crop_id"] == normalized["crop_id"] for item in normalized_existing):
                raise ValueError("duplicate crop_id")

        try:
            append_jsonl_validated(self.path, normalized, validate_existing)
        except _ExistingCropEvent as existing:
            return copy.deepcopy(existing.event)
        return copy.deepcopy(normalized)

    def events(self):
        self._validate_path()
        return copy.deepcopy([
            validate_crop_event(event, require_deterministic_ids=True)
            for event in load_jsonl(self.path, require_terminal_newline=True)
        ])
