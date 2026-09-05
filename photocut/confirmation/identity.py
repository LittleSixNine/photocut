"""Bind confirmation metadata to one immutable finalized detection result."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

from photocut.confirmation.model import _validate_integer_corners
from photocut.confirmation.version import validate_version


@dataclass(frozen=True)
class FinalizedDetectionEvidence:
    """Algorithm identity copied only from a verified finalized image result."""

    algorithm_boundary_corners: tuple[tuple[int, int], ...]
    algorithm_version: str
    detection_id: str | None
    detector_requested: str | None
    detector_used: str | None


def _optional_id(value):
    return value if isinstance(value, str) and value.strip() else None


def _required_id(value):
    return isinstance(value, str) and bool(value.strip())


def match_finalized_detection(
    entry: Mapping,
    reference: Mapping,
    finalized_run: Mapping,
) -> FinalizedDetectionEvidence | None:
    """Return verified per-image evidence, or ``None`` on any mismatch.

    ``entry`` is a mutable ``corners_info.json`` view.  It is used only to
    select one result and to compare its immutable algorithm baseline; all
    algorithm identity returned by this function comes from ``finalized_run``.
    """
    if not all(isinstance(value, Mapping) for value in (entry, reference, finalized_run)):
        return None
    if finalized_run.get("status") != "complete":
        return None

    batch_id = reference.get("batch_id")
    required = ("batch_id", "run_id", "image_id", "source_id")
    if not _required_id(batch_id):
        return None
    if any(not _required_id(entry.get(field)) for field in required):
        return None
    if entry.get("batch_id") != batch_id:
        return None
    reference_run_id = reference.get("run_id")
    if reference_run_id is not None and reference_run_id != entry.get("run_id"):
        return None
    if (
        finalized_run.get("batch_id") != batch_id
        or finalized_run.get("run_id") != entry.get("run_id")
    ):
        return None

    images = finalized_run.get("images")
    if not isinstance(images, list):
        return None
    matches = [
        row for row in images
        if isinstance(row, Mapping)
        and all(row.get(field) == entry.get(field) for field in required)
        and row.get("batch_id") == batch_id
    ]
    if len(matches) != 1:
        return None
    row = matches[0]
    entry_detection_id = entry.get("detection_id")
    if entry_detection_id is not None and row.get("detection_id") != entry_detection_id:
        return None
    try:
        if row.get("detector_requested") == "v8.4":
            # Compare the immutable floating prediction before adapting to the
            # existing integer GUI/annotation contract; never round detection data.
            recorded_raw = row.get("algorithm_boundary_corners")
            mutable_raw = entry.get("algorithm_boundary_corners")
            if recorded_raw != mutable_raw or not isinstance(recorded_raw, (list, tuple)) or len(recorded_raw) != 4:
                return None
            digest = row.get("model_sha256")
            if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest) != 71:
                return None
            if entry.get("model_sha256") != digest or row.get("algorithm_version") != "8.4":
                return None
            for point in recorded_raw:
                if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
                    isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                    for value in point
                ):
                    return None
            recorded = [[int(value) for value in point] for point in recorded_raw]
            mutable = recorded
        else:
            recorded = _validate_integer_corners(
                row.get("algorithm_boundary_corners"), "algorithm_boundary_corners",
            )
            mutable = _validate_integer_corners(
                entry.get("algorithm_boundary_corners"), "algorithm_boundary_corners",
            )
        version = validate_version(row.get("algorithm_version"), "algorithm_version")
    except ValueError:
        return None
    if mutable != recorded:
        return None
    return FinalizedDetectionEvidence(
        tuple(map(tuple, recorded)),
        version,
        _optional_id(row.get("detection_id")),
        _optional_id(row.get("detector_requested")),
        _optional_id(row.get("detector_used")),
    )
