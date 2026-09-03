"""Separate append-only v7 prediction and confirmation feedback log."""
from __future__ import annotations

import copy
import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from photocut.data.dataset_store import append_jsonl_validated, load_jsonl

SCHEMA_VERSION = 1
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[Path, threading.RLock] = {}


def _lock_for(path: Path) -> threading.RLock:
    key = Path(path).resolve(strict=False)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def deterministic_event_id(event: Mapping[str, Any]) -> str:
    """Derive an id from the semantic event, excluding its assigned id."""
    payload = {str(key): value for key, value in event.items() if key != "event_id"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_event(event: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise ValueError("feedback event must be an object")
    normalized = _json_copy(dict(event))
    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("feedback schema_version must be 1")
    event_type = normalized.get("event_type")
    if event_type not in {"prediction", "confirmation"}:
        raise ValueError("feedback event_type is invalid")
    event_id = normalized.get("event_id")
    if not isinstance(event_id, str) or event_id != deterministic_event_id(normalized):
        raise ValueError("feedback event_id is not deterministic")
    for field in ("detection_id", "image_id", "source_hash", "orientation_transform", "algorithm_version", "mode"):
        if not isinstance(normalized.get(field), str) or not normalized[field]:
            raise ValueError(f"feedback {field} must be non-empty")
    if event_type == "confirmation":
        if normalized.get("transaction_phase") not in {"PREPARED", "ANNOTATION_COMMITTED", "FEEDBACK_COMMITTED", "ABORTED"}:
            raise ValueError("invalid confirmation transaction phase")
        if not isinstance(normalized.get("annotation_id"), str) or not normalized["annotation_id"]:
            raise ValueError("confirmation annotation_id is required")
    if "created_at" in normalized and not isinstance(normalized["created_at"], str):
        raise ValueError("created_at must be a string")
    return normalized


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _corners(value: Any) -> list[list[float]] | list[list[int]] | None:
    if value is None:
        return None
    return [[float(point[0]), float(point[1])] for point in value]


def _displayed(value: Any) -> list[list[int]] | None:
    if value is None:
        return None
    return [[int(point[0]), int(point[1])] for point in value]


def build_prediction_event(result: Any, *, displayed_corners: Any = None,
                           alternate_displayed_corners: Any = None,
                           annotation_id: str | None = None,
                           operation: str = "prediction") -> dict[str, Any]:
    identity = result.identity.to_dict()
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_type": "prediction",
        "event_id": "",
        "created_at": _now(),
        "detection_id": identity["request_id"],
        "image_id": identity["image_id"],
        "source_hash": identity["image_id"],
        "orientation_transform": identity["orientation_transform"],
        "algorithm_version": identity["algorithm_version"],
        "parameter_sha256": identity["parameter_sha256"],
        "mode": identity["mode"],
        "status": result.status.value,
        "top1_corners": _corners(result.corners),
        "alternate_corners": _corners(result.alternate_corners),
        "displayed_corners": _displayed(displayed_corners),
        "alternate_displayed_corners": _displayed(alternate_displayed_corners),
        "candidate_sources": list(result.top1_sources),
        "alternate_sources": list(result.alternate_sources),
        "overall_confidence": result.overall_confidence,
        "edge_confidences": list(result.edge_confidences),
        "corner_confidences": list(result.corner_confidences),
        "risks": list(result.risks),
        "timings_ms": dict(result.timings_ms),
        "candidate_audit": [item.to_dict() for item in result.candidate_audit],
        "annotation_id": annotation_id,
        "operation": operation,
        "adjusted_corner_indices": [],
        "corner_movement_px": [],
        "edge_movement_px": [],
        "cancellation_reason": None,
    }
    event["event_id"] = deterministic_event_id(event)
    return event


class FeedbackStore:
    """Thread-safe idempotent JSONL store independent from legacy stores."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = _lock_for(self.path)

    def append(self, event: Mapping[str, Any]) -> dict[str, Any]:
        normalized = _validate_event(event)
        with self._lock:
            def validate_existing(existing):
                for raw in existing:
                    _validate_event(raw)
                    if raw.get("event_id") == normalized["event_id"]:
                        raise ValueError("duplicate feedback event_id")
            append_jsonl_validated(self.path, normalized, validate_existing)
        return copy.deepcopy(normalized)

    def append_idempotent(self, event: Mapping[str, Any]) -> dict[str, Any]:
        normalized = _validate_event(event)
        existing_match = None
        with self._lock:
            existing = load_jsonl(self.path)
            for raw in existing:
                recorded = _validate_event(raw)
                if recorded["event_id"] == normalized["event_id"]:
                    if recorded != normalized:
                        raise ValueError("conflicting duplicate feedback event_id")
                    return copy.deepcopy(recorded)
            def validate_existing(existing):
                for raw in existing:
                    _validate_event(raw)
            if existing_match is None:
                append_jsonl_validated(self.path, normalized, validate_existing)
        return copy.deepcopy(existing_match or normalized)

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy([_validate_event(item) for item in load_jsonl(self.path)])

    def events_for_detection(self, detection_id: str) -> list[dict[str, Any]]:
        return [event for event in self.events() if event.get("detection_id") == detection_id]


__all__ = ["SCHEMA_VERSION", "FeedbackStore", "build_prediction_event", "deterministic_event_id"]
