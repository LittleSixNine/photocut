"""Recoverable binding between a v7 prediction and a legacy annotation."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from photocut.data.annotation_store import AnnotationStore, _validate_event
from photocut.confirmation.model import build_annotation_event

from .feedback_store import FeedbackStore, deterministic_event_id


class ConfirmationConflictError(ValueError):
    """Durable annotation does not match the prepared v7 binding."""


def _snapshot(value: Any) -> list[list[int]]:
    return [[int(point[0]), int(point[1])] for point in value]


def _same_corners(left: Any, right: Any) -> bool:
    try:
        return _snapshot(left) == _snapshot(right)
    except (TypeError, ValueError, IndexError):
        return False


class V7ConfirmationTransaction:
    def __init__(self, feedback_store: FeedbackStore, annotation_store: AnnotationStore,
                 identity: Mapping[str, Any]):
        self.feedback = feedback_store
        self.annotations = annotation_store
        required = ("detection_id", "image_id", "source_hash", "orientation_transform",
                    "algorithm_version", "parameter_sha256", "mode")
        missing = [key for key in required if not isinstance(identity.get(key), str) or not identity[key]]
        if missing:
            raise ValueError(f"missing v7 binding identity: {', '.join(missing)}")
        self.identity = dict(identity)

    def _event(self, phase: str, annotation_id: str, algorithm: Any, boundary: Any,
               *, run_id: str) -> dict[str, Any]:
        event = {
            "schema_version": 1,
            "event_type": "confirmation",
            "event_id": "",
            "detection_id": self.identity["detection_id"],
            "image_id": self.identity["image_id"],
            "source_hash": self.identity["source_hash"],
            "orientation_transform": self.identity["orientation_transform"],
            "algorithm_version": self.identity["algorithm_version"],
            "parameter_sha256": self.identity["parameter_sha256"],
            "mode": self.identity["mode"],
            "transaction_phase": phase,
            "annotation_id": annotation_id,
            "run_id": run_id,
            "algorithm_boundary_corners": _snapshot(algorithm),
            "boundary_corners": _snapshot(boundary),
            "operation": "confirmation",
            "adjusted_corner_indices": [],
            "corner_movement_px": [],
            "edge_movement_px": [],
            "cancellation_reason": None,
        }
        event["event_id"] = deterministic_event_id(event)
        return event

    def prepare(self, algorithm_corners: Any, boundary_corners: Any,
                adjusted_corner_indices: Any, *, run_id: str,
                annotation_id: str | None = None,
                confirmation_duration_ms: int = 0) -> dict[str, Any]:
        annotation = build_annotation_event(
            self.identity["image_id"], run_id, algorithm_corners, boundary_corners,
            adjusted_corner_indices, confirmation_duration_ms,
        )
        if annotation_id is not None:
            annotation["annotation_id"] = annotation_id
        prepared = self._event("PREPARED", annotation["annotation_id"], algorithm_corners,
                               boundary_corners, run_id=run_id)
        prepared["annotation_event"] = copy.deepcopy(annotation)
        # Recompute after embedding the annotation snapshot; recovery can use
        # this record without consulting mutable GUI state.
        prepared["event_id"] = deterministic_event_id(prepared)
        # PREPARED is the first durable boundary.  Calling prepare directly is
        # therefore recoverable; commit may safely retry this append.
        self.feedback.append_idempotent(prepared)
        return prepared

    def commit(self, algorithm_corners: Any, boundary_corners: Any,
               adjusted_corner_indices: Any, *, run_id: str,
               annotation_id: str | None = None,
               confirmation_duration_ms: int = 0) -> dict[str, Any]:
        prepared = self.prepare(
            algorithm_corners, boundary_corners, adjusted_corner_indices,
            run_id=run_id, annotation_id=annotation_id,
            confirmation_duration_ms=confirmation_duration_ms,
        )
        annotation = prepared["annotation_event"]
        self.feedback.append_idempotent(prepared)
        recorded = self.annotations.append_idempotent(annotation)
        committed = self._event("ANNOTATION_COMMITTED", recorded["annotation_id"],
                                recorded["algorithm_boundary_corners"],
                                recorded["boundary_corners"], run_id=recorded["run_id"])
        self.feedback.append_idempotent(committed)
        self._assert_matches(prepared, recorded)
        final = self._event("FEEDBACK_COMMITTED", recorded["annotation_id"],
                            recorded["algorithm_boundary_corners"],
                            recorded["boundary_corners"], run_id=recorded["run_id"])
        self.feedback.append_idempotent(final)
        return copy.deepcopy(recorded)

    def _assert_matches(self, prepared: Mapping[str, Any], annotation: Mapping[str, Any]) -> None:
        if (annotation.get("annotation_id") != prepared.get("annotation_id")
                or annotation.get("image_id") != prepared.get("image_id")
                or not _same_corners(annotation.get("algorithm_boundary_corners"), prepared.get("algorithm_boundary_corners"))
                or not _same_corners(annotation.get("boundary_corners"), prepared.get("boundary_corners"))):
            raise ConfirmationConflictError("annotation does not match prepared v7 binding")

    def recover(self, annotation_id: str) -> dict[str, Any]:
        events = [event for event in self.feedback.events()
                  if event.get("event_type") == "confirmation"
                  and event.get("annotation_id") == annotation_id]
        if not events:
            raise ConfirmationConflictError("no prepared v7 transaction")
        prepared = next((event for event in events if event.get("transaction_phase") == "PREPARED"), None)
        if prepared is None:
            return events[-1]
        annotation_events = self.annotations.events()
        recorded = next((event for event in annotation_events
                          if event.get("annotation_id") == annotation_id), None)
        if recorded is None:
            # A same-run annotation for another source is evidence of a stale
            # or conflicting binding, not a clean absence to silently abort.
            for candidate in annotation_events:
                if (candidate.get("run_id") == prepared.get("run_id")
                        and candidate.get("image_id") != prepared.get("image_id")
                        and _same_corners(candidate.get("boundary_corners"), prepared.get("boundary_corners"))):
                    raise ConfirmationConflictError("annotation source conflicts with prepared v7 binding")
        if recorded is None:
            # A same-run annotation with another ID is a conflicting durable
            # commit, not evidence that the prepared transaction is missing.
            recorded = next((event for event in self.annotations.events()
                             if event.get("run_id") == prepared.get("run_id")), None)
            if recorded is not None:
                self._assert_matches(prepared, {**recorded, "source_hash": self.identity["source_hash"]})
            aborted = dict(prepared)
            aborted["transaction_phase"] = "ABORTED"
            aborted["event_id"] = deterministic_event_id(aborted)
            return self.feedback.append_idempotent(aborted)
        self._assert_matches(prepared, {
            **recorded,
            "source_hash": self.identity["source_hash"],
        })
        committed = self._event("ANNOTATION_COMMITTED", annotation_id,
                                recorded["algorithm_boundary_corners"], recorded["boundary_corners"],
                                run_id=recorded["run_id"])
        self.feedback.append_idempotent(committed)
        final = self._event("FEEDBACK_COMMITTED", annotation_id,
                            recorded["algorithm_boundary_corners"], recorded["boundary_corners"],
                            run_id=recorded["run_id"])
        return self.feedback.append_idempotent(final)


def can_crop_v7(feedback_store: FeedbackStore, identity: Mapping[str, Any],
                annotation_id: str, boundary_corners: Any) -> bool:
    """Return true only for an exact durable FEEDBACK_COMMITTED binding."""
    expected = _snapshot(boundary_corners)
    for event in feedback_store.events():
        if event.get("event_type") != "confirmation" or event.get("transaction_phase") != "FEEDBACK_COMMITTED":
            continue
        if (event.get("annotation_id") == annotation_id
                and all(event.get(key) == identity.get(key) for key in (
                    "detection_id", "image_id", "source_hash", "orientation_transform",
                    "algorithm_version", "parameter_sha256", "mode"))
                and event.get("boundary_corners") == expected):
            return True
    return False


__all__ = ["V7ConfirmationTransaction", "ConfirmationConflictError", "can_crop_v7"]
