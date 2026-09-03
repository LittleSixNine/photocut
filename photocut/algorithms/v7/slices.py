"""Audited, metadata-only evaluation slice derivation.

This module accepts already validated labels/truth and never opens image paths,
loads frozen manifests, or invokes a detector.  Slice labels intentionally
overlap; ``origin_group_id`` remains the independent statistical unit.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _quad(value: Any) -> tuple[tuple[float, float], ...] | None:
    try:
        points = tuple((float(p[0]), float(p[1])) for p in value)
        if len(points) != 4 or len(set(points)) != 4:
            return None
        return points
    except (TypeError, ValueError, IndexError):
        return None


def _edge_lengths(quad):
    return [math.hypot(quad[(i + 1) % 4][0] - quad[i][0], quad[(i + 1) % 4][1] - quad[i][1]) for i in range(4)]


def _turns(quad):
    return [
        (quad[(i + 1) % 4][0] - quad[i][0]) * (quad[(i + 2) % 4][1] - quad[(i + 1) % 4][1])
        - (quad[(i + 1) % 4][1] - quad[i][1]) * (quad[(i + 2) % 4][0] - quad[(i + 1) % 4][0])
        for i in range(4)
    ]


def derive_slices(image: Any = None, audited_labels: Mapping[str, Any] | Sequence[str] | None = None,
                  truth: Any = None, *, image_size: Sequence[int] | None = None,
                  image_id: str | None = None, origin_group_id: str | None = None) -> dict[str, Any]:
    labels: set[str] = set()
    if isinstance(audited_labels, Mapping):
        raw = audited_labels.get("slices", ())
        labels.update(str(item) for item in (raw if isinstance(raw, (list, tuple, set)) else (raw,)))
        labels.update(str(key) for key, value in audited_labels.items() if key != "slices" and value is True)
    elif isinstance(audited_labels, (list, tuple, set)):
        labels.update(str(item) for item in audited_labels)
    quad = _quad(truth)
    if quad:
        lengths = _edge_lengths(quad)
        if max(lengths) > 0 and max(lengths) / max(1e-9, min(lengths)) > 1.02:
            # This is only a weak geometric cue; explicit audited labels remain
            # authoritative for perspective/aspect cases.
            labels.add("perspective")
        turns = _turns(quad)
        if turns and max(abs(value) for value in turns) > 0 and any(value < 0 for value in turns) and any(value > 0 for value in turns):
            labels.add("complex_texture")
        if image_size and len(image_size) == 2:
            width, height = float(image_size[0]), float(image_size[1])
            if any(abs(x) <= 1 or abs(y) <= 1 or abs(x - (width - 1)) <= 1 or abs(y - (height - 1)) <= 1 for x, y in quad):
                labels.add("touches_border")
    if isinstance(audited_labels, Mapping):
        angle = audited_labels.get("rotation_degrees", audited_labels.get("rotation"))
        try:
            if abs(float(angle)) > 1e-6:
                labels.add("rotated")
        except (TypeError, ValueError):
            pass
    return {
        "image_id": image_id,
        "origin_group_id": origin_group_id,
        "slices": tuple(sorted(labels)),
        "independent_unit": origin_group_id or image_id,
    }


def slice_membership(sample: Mapping[str, Any], *, image_size: Sequence[int] | None = None) -> dict[str, Any]:
    return derive_slices(
        audited_labels=sample.get("audited_labels"), truth=sample.get("photo_truth"),
        image_size=image_size, image_id=sample.get("image_id"),
        origin_group_id=sample.get("origin_group_id"),
    )


__all__ = ["derive_slices", "slice_membership"]
