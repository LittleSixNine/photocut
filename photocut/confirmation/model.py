import math
import operator
import stat
import uuid
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


VALID_ZOOM_LEVELS = (2, 4, 8, 16)


@dataclass
class ConfirmationState:
    algorithm_corners: list[list[int]]
    work_corners: list[list[int]]
    image_size: tuple[int, int]
    selected: int = -1
    zoom: int = 4
    adjusted_corner_indices: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.adjusted_corner_indices = {
            index
            for index, (algorithm, work) in enumerate(
                zip(self.algorithm_corners, self.work_corners)
            )
            if work != algorithm
        }

    def move_selected(self, dx: int, dy: int) -> None:
        if not 0 <= self.selected < 4:
            raise ValueError("select a corner before moving")
        self.work_corners[self.selected] = move_corner(
            self.work_corners[self.selected], dx, dy, self.image_size
        )
        if self.work_corners[self.selected] == self.algorithm_corners[self.selected]:
            self.adjusted_corner_indices.discard(self.selected)
        else:
            self.adjusted_corner_indices.add(self.selected)

    def reset(self) -> None:
        self.work_corners = [point[:] for point in self.algorithm_corners]
        self.adjusted_corner_indices.clear()

    def set_zoom(self, zoom: int) -> None:
        if zoom not in VALID_ZOOM_LEVELS:
            raise ValueError(f"zoom must be one of {VALID_ZOOM_LEVELS}")
        self.zoom = zoom


def legacy_preview_corners(
    corners: Sequence[Sequence[int]],
    original_size: tuple[int, int],
    preview_size: tuple[int, int],
) -> list[list[int]]:
    """Derive legacy preview metadata using its historical axis scales and truncation."""
    original_width, original_height = original_size
    preview_width, preview_height = preview_size
    scale_x = preview_width / original_width
    scale_y = preview_height / original_height
    return [
        [int(x * scale_x), int(y * scale_y)]
        for x, y in corners
    ]


def _finite_pair(values: Sequence[Real], name: str) -> tuple[Real, Real]:
    try:
        if len(values) != 2:
            raise ValueError
        first, second = values
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain exactly two finite numbers") from None
    if any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        for value in (first, second)
    ):
        raise ValueError(f"{name} must contain exactly two finite numbers")
    return first, second


def _integer(value, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer") from None


def _non_empty_id(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_integer_corners(values, name: str) -> list[list[int]]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError(f"{name} must contain exactly four 2D integer points")
    corners = []
    for point in values:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name} must contain exactly four 2D integer points")
        try:
            x = _integer(point[0], name)
            y = _integer(point[1], name)
        except ValueError:
            raise ValueError(
                f"{name} must contain exactly four 2D integer points"
            ) from None
        corners.append([x, y])
    return corners


def _validate_adjusted_indices(values) -> list[int]:
    try:
        raw_indices = list(values)
    except TypeError:
        raise ValueError(
            "adjusted_corner_indices must contain unique integers from 0 to 3"
        ) from None
    indices = []
    for value in raw_indices:
        try:
            index = _integer(value, "adjusted_corner_indices")
        except ValueError:
            raise ValueError(
                "adjusted_corner_indices must contain unique integers from 0 to 3"
            ) from None
        if not 0 <= index <= 3:
            raise ValueError(
                "adjusted_corner_indices must contain unique integers from 0 to 3"
            )
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(
            "adjusted_corner_indices must contain unique integers from 0 to 3"
        )
    return sorted(indices)


def build_annotation_event(
    image_id,
    run_id,
    algorithm_boundary_corners,
    boundary_corners,
    adjusted_corner_indices,
    confirmation_duration_ms,
    supersedes_annotation_id=None,
    *,
    schema_version=1,
    gui_version=None,
    algorithm_version=None,
    detection_id=None,
    detector_requested=None,
    detector_used=None,
    confirmation_primary_candidate_id=None,
    selected_candidate_id=None,
    selected_candidate_algorithm_version=None,
):
    image_id = _non_empty_id(image_id, "image_id")
    run_id = _non_empty_id(run_id, "run_id")
    algorithm = _validate_integer_corners(
        algorithm_boundary_corners, "algorithm_boundary_corners"
    )
    boundary = _validate_integer_corners(boundary_corners, "boundary_corners")
    adjusted = _validate_adjusted_indices(adjusted_corner_indices)
    try:
        duration = _integer(confirmation_duration_ms, "confirmation_duration_ms")
    except ValueError:
        raise ValueError(
            "confirmation_duration_ms must be a non-negative integer"
        ) from None
    if duration < 0:
        raise ValueError("confirmation_duration_ms must be a non-negative integer")

    changed = [
        index
        for index, (before, after) in enumerate(zip(algorithm, boundary))
        if before != after
    ]
    if adjusted != changed:
        raise ValueError(
            "adjusted_corner_indices must exactly match changed corners"
        )

    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise ValueError("schema_version must be integer 1 or 2")
    identity = {}
    if schema_version == 1:
        if any(
            value is not None
            for value in (
                gui_version,
                algorithm_version,
                detection_id,
                detector_requested,
                detector_used,
                confirmation_primary_candidate_id,
                selected_candidate_id,
                selected_candidate_algorithm_version,
            )
        ):
            raise ValueError("schema v1 cannot contain GUI or detector identity")
    else:
        from photocut.confirmation.version import validate_version

        identity["gui_version"] = validate_version(gui_version, "gui_version")
        if algorithm_version is not None:
            identity["algorithm_version"] = validate_version(
                algorithm_version, "algorithm_version"
            )
        for field, value in (
            ("detection_id", detection_id),
            ("detector_requested", detector_requested),
            ("detector_used", detector_used),
            (
                "confirmation_primary_candidate_id",
                confirmation_primary_candidate_id,
            ),
            ("selected_candidate_id", selected_candidate_id),
        ):
            if value is not None:
                identity[field] = _non_empty_id(value, field)
        if selected_candidate_algorithm_version is not None:
            identity["selected_candidate_algorithm_version"] = validate_version(
                selected_candidate_algorithm_version,
                "selected_candidate_algorithm_version",
            )

    annotation_id = f"ann_{uuid.uuid4().hex}"
    if supersedes_annotation_id is not None:
        supersedes_annotation_id = _non_empty_id(
            supersedes_annotation_id, "supersedes_annotation_id"
        )
        revision_identity = {
            "schema_version": schema_version,
            "image_id": image_id,
            "run_id": run_id,
            "algorithm_boundary_corners": algorithm,
            "boundary_corners": boundary,
            "supersedes_annotation_id": supersedes_annotation_id,
            **{field: identity[field] for field in sorted(identity)},
        }
        digest = hashlib.sha256(
            json.dumps(
                revision_identity, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        annotation_id = f"ann_rev_{digest}"

    event = {
        "schema_version": schema_version,
        "annotation_id": annotation_id,
        "image_id": image_id,
        "run_id": run_id,
        "algorithm_boundary_corners": algorithm,
        "boundary_corners": boundary,
        "confirmation": "adjusted" if changed else "accepted",
        "adjusted_corner_indices": changed,
        "corner_errors_px": [
            round(math.hypot(after[0] - before[0], after[1] - before[1]), 3)
            for before, after in zip(algorithm, boundary)
        ],
        "confirmation_duration_ms": duration,
        "confirmed_at": datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
    }
    if schema_version == 2:
        event.update(identity)
    if supersedes_annotation_id is not None:
        event["supersedes_annotation_id"] = supersedes_annotation_id
    return event


def build_crop_event(entry: Mapping, output_path, inset: Mapping, *, error=None):
    """Build a validated crop-attempt event without mutating UI metadata."""
    from photocut.data.crop_event_store import (
        _validate_numeric_corners,
        deterministic_event_id,
        validate_crop_event,
    )

    image_id = _non_empty_id(entry.get("image_id"), "image_id")
    annotation_id = _non_empty_id(entry.get("annotation_id"), "annotation_id")
    boundary = _validate_integer_corners(
        entry.get("boundary_corners"), "boundary_corners"
    )
    crop = _validate_numeric_corners(entry.get("crop_corners"), "crop_corners")
    if not isinstance(inset, Mapping) or inset.get("method") != "parallel_edge_offset":
        raise ValueError("invalid crop inset method")
    distance = inset.get("distance_px")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, Real)
        or not math.isfinite(distance)
        or distance < 0
    ):
        raise ValueError("invalid crop inset distance")
    status = "failed" if error is not None else "succeeded"
    output = None
    if status == "succeeded":
        output = Path(output_path)
        try:
            output_info = output.lstat()
        except OSError:
            output_info = None
        if output_info is None or stat.S_ISLNK(output_info.st_mode) or not stat.S_ISREG(output_info.st_mode):
            raise ValueError("crop output path does not exist")
        output = str(output)
    elif not isinstance(error, str) or not error.strip():
        raise ValueError("failed crop event requires non-empty error")
    event = {
        "schema_version": 1,
        "event_id": "",
        "crop_id": "",
        "image_id": image_id,
        "annotation_id": annotation_id,
        "boundary_corners": boundary,
        "crop_corners": crop,
        "inset": {"method": "parallel_edge_offset", "distance_px": float(distance)},
        "output_path": output,
        "status": status,
        "error": error,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    event_id = deterministic_event_id(event)
    event["event_id"] = event_id
    event["crop_id"] = event_id
    return validate_crop_event(event)


@dataclass(frozen=True)
class DisplayTransform:
    scale: float
    offset_x: int
    offset_y: int
    original_width: int
    original_height: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, Real)
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("scale must be finite and positive")
        _finite_pair((self.offset_x, self.offset_y), "offset")
        width = _integer(self.original_width, "original_width")
        height = _integer(self.original_height, "original_height")
        if width <= 0 or height <= 0:
            raise ValueError("original dimensions must be positive")


def original_to_display(
    point: Sequence[Real], transform: DisplayTransform
) -> tuple[int, int]:
    x, y = _finite_pair(point, "point")
    return (
        round(x * transform.scale + transform.offset_x),
        round(y * transform.scale + transform.offset_y),
    )


def display_to_original(
    point: Sequence[Real], transform: DisplayTransform
) -> list[int]:
    x, y = _finite_pair(point, "point")
    original_x = round((x - transform.offset_x) / transform.scale)
    original_y = round((y - transform.offset_y) / transform.scale)
    return move_corner(
        [original_x, original_y],
        0,
        0,
        (transform.original_width, transform.original_height),
    )


def move_corner(
    corner: Sequence[int], dx: int, dy: int, image_size: tuple[int, int]
) -> list[int]:
    corner_x, corner_y = _finite_pair(corner, "corner")
    corner_x = _integer(corner_x, "corner")
    corner_y = _integer(corner_y, "corner")
    try:
        width_value, height_value = image_size
        if len(image_size) != 2:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("image_size must contain exactly two positive integers") from None
    width = _integer(width_value, "image_size")
    height = _integer(height_value, "image_size")
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain exactly two positive integers")
    try:
        delta_x = _integer(dx, "dx")
        delta_y = _integer(dy, "dy")
    except ValueError:
        raise ValueError("dx and dy must be integers") from None
    return [
        max(0, min(width - 1, corner_x + delta_x)),
        max(0, min(height - 1, corner_y + delta_y)),
    ]


def _zoom_level(value) -> int:
    try:
        zoom = _integer(value, "zoom")
    except ValueError:
        raise ValueError(f"zoom must be one of {VALID_ZOOM_LEVELS}") from None
    if zoom not in VALID_ZOOM_LEVELS:
        raise ValueError(f"zoom must be one of {VALID_ZOOM_LEVELS}")
    return zoom


def render_magnifier_source(
    image: np.ndarray,
    center: Sequence[int],
    zoom: int,
    viewport_size: int = 400,
) -> np.ndarray:
    """Return unscaled original pixels, black-padding any area outside the image."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim not in (2, 3)
        or any(dimension <= 0 for dimension in image.shape)
    ):
        raise ValueError("image must be a non-empty 2D or 3D array")

    zoom = _zoom_level(zoom)
    try:
        viewport_size = _integer(viewport_size, "viewport_size")
    except ValueError:
        raise ValueError("viewport_size must be a positive integer") from None
    if viewport_size <= 0 or viewport_size % zoom != 0:
        raise ValueError("viewport_size must be positive and divisible by zoom")

    try:
        if len(center) != 2:
            raise ValueError
        center_x = _integer(center[0], "center")
        center_y = _integer(center[1], "center")
    except (TypeError, ValueError):
        raise ValueError("center must contain exactly two integers") from None

    image_height, image_width = image.shape[:2]
    if not (0 <= center_x < image_width and 0 <= center_y < image_height):
        raise ValueError("center must be within image bounds")

    source_size = viewport_size // zoom
    half = source_size // 2
    left = center_x - half
    top = center_y - half
    right = left + source_size
    bottom = top + source_size

    image_left = max(0, left)
    image_top = max(0, top)
    image_right = min(image_width, right)
    image_bottom = min(image_height, bottom)
    source_left = image_left - left
    source_top = image_top - top
    source_right = source_left + image_right - image_left
    source_bottom = source_top + image_bottom - image_top

    source = np.zeros(
        (source_size, source_size, *image.shape[2:]),
        dtype=image.dtype,
    )
    source[source_top:source_bottom, source_left:source_right] = image[
        image_top:image_bottom, image_left:image_right
    ]
    return source


def magnifier_drag_delta(
    screen_dx: int, screen_dy: int, zoom: int
) -> tuple[int, int]:
    """Map inverse screen drag using Python round's ties-to-even rule."""
    zoom = _zoom_level(zoom)
    try:
        screen_dx = _integer(screen_dx, "screen_dx")
        screen_dy = _integer(screen_dy, "screen_dy")
    except ValueError:
        raise ValueError("screen_dx and screen_dy must be integers") from None
    return (-round(screen_dx / zoom), -round(screen_dy / zoom))


class MagnifierDragCapture:
    """Window-independent magnifier drag state with subpixel residuals."""

    def __init__(self) -> None:
        self.active = False
        self._anchor_x = 0.0
        self._anchor_y = 0.0

    @staticmethod
    def _sample_values(sample):
        try:
            x = float(sample.x)
            y = float(sample.y)
            left_down = bool(sample.left_down)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("invalid pointer sample") from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("pointer coordinates must be finite")
        return x, y, left_down

    def begin(self, sample) -> None:
        try:
            x, y, left_down = self._sample_values(sample)
        except ValueError:
            self.cancel()
            raise
        if not left_down:
            self.cancel()
            return
        self._anchor_x = x
        self._anchor_y = y
        self.active = True

    def update(self, sample, *, zoom: int) -> tuple[int, int]:
        if not self.active:
            return (0, 0)
        try:
            x, y, left_down = self._sample_values(sample)
        except ValueError as exc:
            self.cancel()
            from photocut.confirmation.pointer import PointerReadError
            raise PointerReadError(str(exc)) from exc
        if not left_down:
            self.cancel()
            return (0, 0)
        try:
            zoom = _zoom_level(zoom)
            source_dx, source_dy = magnifier_drag_delta(
                int(round(x - self._anchor_x)),
                int(round(y - self._anchor_y)),
                zoom,
            )
        except (TypeError, ValueError) as exc:
            self.cancel()
            from photocut.confirmation.pointer import PointerReadError
            raise PointerReadError(str(exc)) from exc
        # Advance by consumed screen distance, not by the current pointer.
        # This keeps the signed residual when screen motion is not a zoom
        # multiple and prevents small movements from being lost.
        self._anchor_x += -source_dx * zoom
        self._anchor_y += -source_dy * zoom
        return source_dx, source_dy

    def cancel(self) -> None:
        self.active = False
        self._anchor_x = 0.0
        self._anchor_y = 0.0
