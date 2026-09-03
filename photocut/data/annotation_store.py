import copy
import math
import re
import threading
from datetime import datetime
from numbers import Real
from pathlib import Path

from photocut.confirmation.model import (
    _non_empty_id,
    _validate_adjusted_indices,
    _validate_integer_corners,
    build_annotation_event,
)
from photocut.data.dataset_store import append_jsonl_validated, load_jsonl
from photocut.confirmation.version import validate_version


_BASE_REQUIRED_FIELDS = {
    "schema_version",
    "annotation_id",
    "image_id",
    "run_id",
    "algorithm_boundary_corners",
    "boundary_corners",
    "confirmation",
    "adjusted_corner_indices",
    "corner_errors_px",
    "confirmation_duration_ms",
    "confirmed_at",
}
_V1_OPTIONAL_FIELDS = {"supersedes_annotation_id", "provenance"}
_V2_REQUIRED_FIELDS = _BASE_REQUIRED_FIELDS | {"gui_version"}
_V2_OPTIONAL_FIELDS = {
    "supersedes_annotation_id",
    "algorithm_version",
    "detection_id",
    "detector_requested",
    "detector_used",
    "confirmation_primary_candidate_id",
    "selected_candidate_id",
    "selected_candidate_algorithm_version",
}
_VOLATILE_RETRY_FIELDS = {
    "annotation_id",
    "confirmed_at",
    "confirmation_duration_ms",
}
_LOCKS_GUARD = threading.Lock()
_LOCKS = {}


class _ExistingAnnotation(Exception):
    pass


def _lock_for(path: Path) -> threading.RLock:
    path = Path(path)
    key = path.parent.resolve(strict=False) / path.name
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _validate_event(event) -> dict:
    if not isinstance(event, dict):
        raise ValueError("annotation event schema must be an object")
    schema_version = event.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("schema_version must be integer 1 or 2")
    required_fields = (
        _BASE_REQUIRED_FIELDS if schema_version == 1 else _V2_REQUIRED_FIELDS
    )
    optional_fields = (
        _V1_OPTIONAL_FIELDS if schema_version == 1 else _V2_OPTIONAL_FIELDS
    )
    fields = set(event)
    if not required_fields.issubset(fields) or not fields.issubset(
        required_fields | optional_fields
    ):
        raise ValueError("annotation event schema has missing or unexpected fields")

    annotation_id = _non_empty_id(event["annotation_id"], "annotation_id")
    image_id = _non_empty_id(event["image_id"], "image_id")
    run_id = _non_empty_id(event["run_id"], "run_id")
    algorithm = _validate_integer_corners(
        event["algorithm_boundary_corners"], "algorithm_boundary_corners"
    )
    boundary = _validate_integer_corners(
        event["boundary_corners"], "boundary_corners"
    )
    adjusted = _validate_adjusted_indices(event["adjusted_corner_indices"])
    changed = [
        index
        for index, (before, after) in enumerate(zip(algorithm, boundary))
        if before != after
    ]
    if adjusted != changed:
        raise ValueError(
            "adjusted_corner_indices must exactly match changed corners"
        )

    expected_confirmation = "adjusted" if changed else "accepted"
    if event["confirmation"] != expected_confirmation:
        raise ValueError("confirmation must be derived from boundary coordinates")
    expected_errors = [
        round(math.hypot(after[0] - before[0], after[1] - before[1]), 3)
        for before, after in zip(algorithm, boundary)
    ]
    errors = event["corner_errors_px"]
    if (
        not isinstance(errors, (list, tuple))
        or len(errors) != 4
        or any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            for value in errors
        )
        or list(errors) != expected_errors
    ):
        raise ValueError("corner_errors_px must be derived from boundary coordinates")

    duration = event["confirmation_duration_ms"]
    if type(duration) is not int or duration < 0:
        raise ValueError("confirmation_duration_ms must be a non-negative integer")
    confirmed_at = event["confirmed_at"]
    if not isinstance(confirmed_at, str) or not confirmed_at:
        raise ValueError("confirmed_at must be a timezone-aware ISO timestamp")
    try:
        parsed_time = datetime.fromisoformat(confirmed_at)
    except ValueError:
        raise ValueError(
            "confirmed_at must be a timezone-aware ISO timestamp"
        ) from None
    if parsed_time.utcoffset() is None:
        raise ValueError("confirmed_at must be a timezone-aware ISO timestamp")

    normalized = {
        "schema_version": schema_version,
        "annotation_id": annotation_id,
        "image_id": image_id,
        "run_id": run_id,
        "algorithm_boundary_corners": algorithm,
        "boundary_corners": boundary,
        "confirmation": expected_confirmation,
        "adjusted_corner_indices": adjusted,
        "corner_errors_px": expected_errors,
        "confirmation_duration_ms": duration,
        "confirmed_at": confirmed_at,
    }
    if "supersedes_annotation_id" in event:
        normalized["supersedes_annotation_id"] = _non_empty_id(
            event["supersedes_annotation_id"], "supersedes_annotation_id"
        )
    if schema_version == 2:
        normalized["gui_version"] = validate_version(
            event["gui_version"], "gui_version"
        )
        if "algorithm_version" in event:
            normalized["algorithm_version"] = validate_version(
                event["algorithm_version"], "algorithm_version"
            )
        for field in (
            "detection_id",
            "detector_requested",
            "detector_used",
            "confirmation_primary_candidate_id",
            "selected_candidate_id",
        ):
            if field in event:
                normalized[field] = _non_empty_id(event[field], field)
        if "selected_candidate_algorithm_version" in event:
            normalized["selected_candidate_algorithm_version"] = validate_version(
                event["selected_candidate_algorithm_version"],
                "selected_candidate_algorithm_version",
            )
    elif "provenance" in event:
        provenance = event["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "source_format",
            "source_json_sha256",
            "original_filename",
            "algorithm_result_available",
        }:
            raise ValueError("invalid annotation provenance")
        if provenance["source_format"] not in {
            "legacy_corners_info", "legacy_ground_truth"
        }:
            raise ValueError("invalid annotation provenance")
        source_hash = provenance["source_json_sha256"]
        if not isinstance(source_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", source_hash
        ):
            raise ValueError("invalid annotation provenance")
        filename = provenance["original_filename"]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError("invalid annotation provenance")
        available = provenance["algorithm_result_available"]
        if type(available) is not bool:
            raise ValueError("invalid annotation provenance")
        expected = provenance["source_format"] == "legacy_corners_info"
        if available != expected:
            raise ValueError("invalid annotation provenance")
        normalized["provenance"] = copy.deepcopy(provenance)
    return normalized


def _stable_retry_payload(event: dict) -> dict:
    return {
        key: copy.deepcopy(value)
        for key, value in event.items()
        if key not in _VOLATILE_RETRY_FIELDS
    }


def _reject_cycles(events: list[dict], by_id: dict[str, dict]) -> None:
    states = {}
    for event in events:
        current = event["annotation_id"]
        path = []
        while current is not None and states.get(current, 0) == 0:
            states[current] = 1
            path.append(current)
            current = by_id[current].get("supersedes_annotation_id")
        if current is not None and states[current] == 1:
            raise ValueError("annotation history contains a supersedes cycle")
        for annotation_id in path:
            states[annotation_id] = 2


def _validate_history(raw_events) -> tuple[list[dict], dict[str, str]]:
    events = [_validate_event(event) for event in raw_events]
    by_id = {}
    positions = {}
    for position, event in enumerate(events):
        annotation_id = event["annotation_id"]
        if annotation_id in by_id:
            raise ValueError(f"duplicate annotation_id: {annotation_id}")
        by_id[annotation_id] = event
        positions[annotation_id] = position

    for event in events:
        supersedes = event.get("supersedes_annotation_id")
        if supersedes is None:
            continue
        if supersedes not in by_id:
            raise ValueError(f"unknown supersedes_annotation_id: {supersedes}")
        if by_id[supersedes]["image_id"] != event["image_id"]:
            raise ValueError("supersedes_annotation_id targets another image")
    _reject_cycles(events, by_id)

    active_heads = {}
    for position, event in enumerate(events):
        image_id = event["image_id"]
        annotation_id = event["annotation_id"]
        supersedes = event.get("supersedes_annotation_id")
        if supersedes is None:
            if image_id in active_heads:
                raise ValueError(f"multiple active heads for image_id: {image_id}")
        else:
            if positions[supersedes] >= position:
                raise ValueError(
                    "supersedes_annotation_id must reference an earlier annotation"
                )
            if active_heads.get(image_id) != supersedes:
                raise ValueError(
                    "revision must supersede the current active head without branching"
                )
        active_heads[image_id] = annotation_id
    return events, active_heads


class AnnotationStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = _lock_for(self.path)

    def append(self, event) -> None:
        with self._lock:
            normalized = _validate_event(event)

            def validate_existing(existing):
                _validate_history([*existing, normalized])

            append_jsonl_validated(self.path, normalized, validate_existing)

    def append_idempotent(self, event) -> dict:
        """Append a revision once, returning the recorded event on a retry."""
        with self._lock:
            normalized = _validate_event(event)
            existing_match = None

            def validate_existing(existing):
                nonlocal existing_match
                events, _ = _validate_history(existing)
                for recorded in events:
                    if recorded["annotation_id"] != normalized["annotation_id"]:
                        continue
                    if (
                        normalized.get("supersedes_annotation_id") is None
                        or _stable_retry_payload(recorded)
                        != _stable_retry_payload(normalized)
                    ):
                        raise ValueError("duplicate annotation_id")
                    existing_match = copy.deepcopy(recorded)
                    raise _ExistingAnnotation(recorded)
                _validate_history([*events, normalized])

            try:
                append_jsonl_validated(self.path, normalized, validate_existing)
            except _ExistingAnnotation:
                return existing_match
            return copy.deepcopy(normalized)

    def events(self) -> list[dict]:
        with self._lock:
            events, _ = _validate_history(load_jsonl(self.path))
            return copy.deepcopy(events)

    def latest_by_image(self) -> dict[str, dict]:
        with self._lock:
            events, active_heads = _validate_history(load_jsonl(self.path))
            by_id = {event["annotation_id"]: event for event in events}
            return copy.deepcopy(
                {
                    image_id: by_id[annotation_id]
                    for image_id, annotation_id in active_heads.items()
                }
            )

    def revise(
        self,
        previous,
        boundary_corners,
        adjusted_corner_indices,
        confirmation_duration_ms,
    ) -> dict:
        previous = _validate_event(previous)
        return build_annotation_event(
            image_id=previous["image_id"],
            run_id=previous["run_id"],
            algorithm_boundary_corners=previous["algorithm_boundary_corners"],
            boundary_corners=boundary_corners,
            adjusted_corner_indices=adjusted_corner_indices,
            confirmation_duration_ms=confirmation_duration_ms,
            supersedes_annotation_id=previous["annotation_id"],
        )
