"""Numerically conservative, scale-safe quadrilateral geometry for v7.

The public helpers intentionally use ordinary Python sequences at their boundary and
return tuples.  No helper clips, rounds, or silently repairs malformed geometry.
Coordinates use the image convention (x right, y down), so a TL/TR/BR/BL polygon
has positive signed area.
"""
from __future__ import annotations

from collections.abc import Sequence
import math
import numbers
from typing import Any

import numpy as np


class GeometryError(ValueError):
    """Raised when geometry is non-finite, ambiguous, degenerate, or unsafe."""


Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]


def _raw_points(value: Any, *, count: int | None = 4) -> list[Point]:
    """Validate nested numeric sequences before any array/float conversion."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        # ndarray deliberately gets a small explicit accommodation while still
        # validating every scalar before conversion.
        if hasattr(value, "shape") and hasattr(value, "__getitem__"):
            try:
                value = value.tolist()
            except Exception as exc:  # pragma: no cover - defensive
                raise GeometryError("points must be a numeric sequence") from exc
        else:
            raise GeometryError("points must be a numeric sequence")
    try:
        n = len(value)
    except Exception as exc:
        raise GeometryError("points must be a sized sequence") from exc
    if count is not None and n != count:
        raise GeometryError(f"expected {count} points")
    out: list[Point] = []
    for point in value:
        if isinstance(point, (str, bytes)) or not isinstance(point, Sequence):
            if hasattr(point, "shape") and hasattr(point, "__getitem__"):
                try:
                    point = point.tolist()
                except Exception as exc:  # pragma: no cover
                    raise GeometryError("each point must contain two numbers") from exc
            else:
                raise GeometryError("each point must contain two numbers")
        try:
            if len(point) != 2:
                raise GeometryError("each point must contain two numbers")
            x, y = point[0], point[1]
        except GeometryError:
            raise
        except Exception as exc:
            raise GeometryError("each point must contain two numbers") from exc
        for item in (x, y):
            if isinstance(item, (bool, np.bool_)) or not isinstance(item, numbers.Real):
                raise TypeError("coordinates must be real numbers")
            if not math.isfinite(float(item)):
                raise ValueError("coordinates must be finite")
        # Conversion follows validation; this prevents numpy/object coercion from
        # hiding NaN, infinities, booleans, or malformed nested values.
        out.append((float(x), float(y)))
    return out


def _signed_area_points(points: Sequence[Point]) -> float:
    # Translate to the first vertex before the shoelace sum.  This avoids
    # catastrophic cancellation for otherwise ordinary quads near 1e12.
    ox, oy = points[0]
    local = [(x - ox, y - oy) for x, y in points]
    return 0.5 * sum(
        local[i][0] * local[(i + 1) % len(local)][1]
        - local[(i + 1) % len(local)][0] * local[i][1]
        for i in range(len(local))
    )


def signed_area(points: Sequence[Sequence[float]]) -> float:
    """Return the shoelace signed area without changing point order."""
    raw = _raw_points(points, count=None)
    if len(raw) < 3:
        raise GeometryError("at least three points are required")
    return float(_signed_area_points(raw))


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _validate_convex(ordered: Sequence[Point]) -> None:
    # Use a local coordinate span rather than absolute coordinates.  A quad at
    # (1e9, 1e9) must have exactly the same validity as the same quad at origin.
    span = max(1.0, max(x for x, _ in ordered) - min(x for x, _ in ordered), max(y for _, y in ordered) - min(y for _, y in ordered))
    eps = 1e-12 * span * span
    turns = [_cross(ordered[i], ordered[(i + 1) % 4], ordered[(i + 2) % 4]) for i in range(4)]
    if any(abs(v) <= eps for v in turns):
        raise GeometryError("quadrilateral is degenerate or collinear")
    if not (all(v > eps for v in turns) or all(v < -eps for v in turns)):
        raise GeometryError("quadrilateral must be strictly convex and non-self-intersecting")
    if abs(_signed_area_points(ordered)) <= eps:
        raise GeometryError("quadrilateral area is too small")


def order_quad(corners: Sequence[Sequence[float]]) -> Quad:
    """Order four points as TL, TR, BR, BL using centroid/angle geometry."""
    points = _raw_points(corners, count=4)
    # Distinctness is checked before sorting; exact duplicates are never repaired.
    if len(set(points)) != 4:
        raise GeometryError("quadrilateral contains duplicate points")
    ox, oy = points[0]
    local = [(x - ox, y - oy) for x, y in points]
    cx = sum(x for x, _ in local) / 4.0
    cy = sum(y for _, y in local) / 4.0
    angles = [math.atan2(y - cy, x - cx) for x, y in local]
    if len({round(a, 14) for a in angles}) != 4:
        raise GeometryError("ambiguous centroid/angle ordering")
    ordered = [point for _, point in sorted(zip(angles, points), key=lambda item: item[0])]
    if _signed_area_points(ordered) < 0:
        ordered.reverse()
    # Compare x+y in the same translated frame to retain precision at large
    # absolute coordinates.
    sums = [(x - ox) + (y - oy) for x, y in ordered]
    minimum = min(sums)
    scale = max(1.0, max(abs(v) for v in sums))
    tied = [i for i, value in enumerate(sums) if abs(value - minimum) <= 1e-12 * scale]
    if len(tied) != 1:
        raise GeometryError("ambiguous TL corner: equal x+y")
    start = tied[0]
    ordered = ordered[start:] + ordered[:start]
    # A valid image-coordinate quad must wind TL→TR→BR→BL consistently.
    if _signed_area_points(ordered) <= 0:
        raise GeometryError("invalid quadrilateral winding")
    _validate_convex(ordered)
    return tuple(ordered)  # type: ignore[return-value]


def validate_quad(
    corners: Sequence[Sequence[float]],
    image_size: Sequence[float] | None = None,
    *,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 1.0,
    min_edge_ratio: float = 0.0,
) -> Quad:
    """Order and validate a quad, optionally against image-relative limits."""
    raw = _raw_points(corners, count=4)
    # ``validate_quad`` is the strict API for an already connected polygon: do
    # not silently turn a bow-tie or concave path into its convex hull.  Call
    # ``order_quad`` when points are deliberately supplied in arbitrary order.
    turns = [_cross(raw[i], raw[(i + 1) % 4], raw[(i + 2) % 4]) for i in range(4)]
    if (_segments_intersect(raw[0], raw[1], raw[2], raw[3]) or
            _segments_intersect(raw[1], raw[2], raw[3], raw[0])):
        raise GeometryError("quadrilateral is self-intersecting")
    nonzero = [v for v in turns if abs(v) > 1e-12]
    if len(nonzero) == 4 and not (all(v > 0 for v in nonzero) or all(v < 0 for v in nonzero)):
        raise GeometryError("quadrilateral is concave")
    quad = order_quad(raw)
    if image_size is not None:
        width, height = _size(image_size)
        _check_bounds(quad, width, height)
        ratio = abs(_signed_area_points(quad)) / (width * height)
        if ratio < min_area_ratio or ratio > max_area_ratio:
            raise GeometryError("quadrilateral area ratio is outside bounds")
        if min_normalized_edge(quad, (width, height)) < min_edge_ratio:
            raise GeometryError("quadrilateral has a too-short normalized edge")
    return quad


def _check_bounds(quad: Sequence[Point], width: float, height: float) -> None:
    if any(x < 0 or x > width - 1 or y < 0 or y > height - 1 for x, y in quad):
        raise GeometryError("quadrilateral corner is outside image bounds")


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orient(p: Point, q: Point, r: Point) -> float:
        return _cross(p, q, r)
    ab_c, ab_d = orient(a, b, c), orient(a, b, d)
    cd_a, cd_b = orient(c, d, a), orient(c, d, b)
    span = max(
        1.0,
        max(p[0] for p in (a, b, c, d)) - min(p[0] for p in (a, b, c, d)),
        max(p[1] for p in (a, b, c, d)) - min(p[1] for p in (a, b, c, d)),
    )
    eps = 1e-12 * span * span
    if abs(ab_c) <= eps and min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps:
        return True
    if abs(ab_d) <= eps and min(a[0], b[0]) - eps <= d[0] <= max(a[0], b[0]) + eps and min(a[1], b[1]) - eps <= d[1] <= max(a[1], b[1]) + eps:
        return True
    if abs(cd_a) <= eps and min(c[0], d[0]) - eps <= a[0] <= max(c[0], d[0]) + eps and min(c[1], d[1]) - eps <= a[1] <= max(c[1], d[1]) + eps:
        return True
    if abs(cd_b) <= eps and min(c[0], d[0]) - eps <= b[0] <= max(c[0], d[0]) + eps and min(c[1], d[1]) - eps <= b[1] <= max(c[1], d[1]) + eps:
        return True
    return (ab_c > eps) != (ab_d > eps) and (cd_a > eps) != (cd_b > eps)


def _size(size: Sequence[float]) -> tuple[float, float]:
    if isinstance(size, (str, bytes)) or not isinstance(size, Sequence) or len(size) != 2:
        raise GeometryError("image size must be (width, height)")
    w, h = size
    if isinstance(w, (bool, np.bool_)) or isinstance(h, (bool, np.bool_)) or not isinstance(w, numbers.Real) or not isinstance(h, numbers.Real):
        raise TypeError("image size must contain numbers")
    w, h = float(w), float(h)
    if not math.isfinite(w) or not math.isfinite(h) or w <= 0 or h <= 0:
        raise ValueError("image size must be finite and positive")
    return w, h


def quad_area_ratio(corners: Sequence[Sequence[float]], image_size: Sequence[float]) -> float:
    quad = order_quad(corners)
    width, height = _size(image_size)
    return abs(_signed_area_points(quad)) / (width * height)


def min_normalized_edge(corners: Sequence[Sequence[float]], image_size: Sequence[float]) -> float:
    quad = order_quad(corners)
    width, height = _size(image_size)
    diagonal = math.hypot(width, height)
    lengths = [math.hypot(quad[(i + 1) % 4][0] - quad[i][0], quad[(i + 1) % 4][1] - quad[i][1]) for i in range(4)]
    return min(lengths) / diagonal


def line_intersection(
    p1: Sequence[float], p2: Sequence[float], q1: Sequence[float], q2: Sequence[float], *, tolerance: float = 1e-9
) -> Point:
    """Intersect infinite lines, rejecting parallel/near-parallel directions."""
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, numbers.Real):
        raise GeometryError("tolerance must be a finite non-negative number")
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise GeometryError("tolerance must be a finite non-negative number")
    points = _raw_points((p1, p2, q1, q2), count=4)
    a, b, c, d = points
    r = (b[0] - a[0], b[1] - a[1])
    s = (d[0] - c[0], d[1] - c[1])
    denom = r[0] * s[1] - r[1] * s[0]
    scale = math.hypot(*r) * math.hypot(*s)
    if scale == 0 or abs(denom) <= tolerance * scale:
        raise GeometryError("lines are parallel or near-parallel")
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / denom
    point = (a[0] + t * r[0], a[1] + t * r[1])
    if not all(math.isfinite(v) for v in point):
        raise GeometryError("line intersection is non-finite")
    return point


def _poly_area(poly: Sequence[Point]) -> float:
    return _signed_area_points(poly)


def _clip(subject: list[Point], clipper: Sequence[Point]) -> list[Point]:
    if not subject:
        return []
    output = subject
    for i, edge_start in enumerate(clipper):
        edge_end = clipper[(i + 1) % len(clipper)]
        input_points = output
        output = []
        if not input_points:
            break

        def inside(p: Point) -> bool:
            return _cross(edge_start, edge_end, p) >= -1e-12

        for current, previous in zip(input_points, input_points[-1:] + input_points[:-1]):
            current_inside, previous_inside = inside(current), inside(previous)
            if current_inside != previous_inside:
                # Segment-line intersection; clipping edges are guaranteed nonzero.
                output.append(line_intersection(previous, current, edge_start, edge_end))
            if current_inside:
                output.append(current)
    return output


def polygon_iou(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Return intersection-over-union for convex polygons (quads are ordered)."""
    pa = order_quad(a) if len(a) == 4 else _prepare_convex_polygon(a)
    pb = order_quad(b) if len(b) == 4 else _prepare_convex_polygon(b)
    if len(pa) < 3 or len(pb) < 3:
        raise GeometryError("polygons need at least three points")
    if _poly_area(pa) < 0:
        pa = tuple(reversed(pa))
    if _poly_area(pb) < 0:
        pb = tuple(reversed(pb))
    area_a, area_b = abs(_poly_area(pa)), abs(_poly_area(pb))
    if area_a <= 0 or area_b <= 0:
        raise GeometryError("polygon area must be positive")
    inter = _clip(list(pa), pb)
    inter_area = abs(_poly_area(inter)) if len(inter) >= 3 else 0.0
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _prepare_convex_polygon(value: Sequence[Sequence[float]]) -> tuple[Point, ...]:
    points = _raw_points(value, count=None)
    if len(points) < 3 or len(set(points)) != len(points):
        raise GeometryError("polygon must contain distinct points")
    turns = [_cross(points[i], points[(i + 1) % len(points)], points[(i + 2) % len(points)]) for i in range(len(points))]
    span = max(1.0, max(x for x, _ in points) - min(x for x, _ in points), max(y for _, y in points) - min(y for _, y in points))
    eps = 1e-12 * span * span
    if any(abs(turn) <= eps for turn in turns) or not (all(turn > eps for turn in turns) or all(turn < -eps for turn in turns)):
        raise GeometryError("polygon_iou only supports strictly convex polygons")
    if _signed_area_points(points) < 0:
        points.reverse()
    return tuple(points)


def normalized_corner_distance(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]], image_size: Sequence[float]
) -> float:
    qa, qb = order_quad(a), order_quad(b)
    width, height = _size(image_size)
    diagonal = math.hypot(width, height)
    return float(sum(math.hypot(x1 - x2, y1 - y2) for (x1, y1), (x2, y2) in zip(qa, qb)) / 4.0 / diagonal)


def _scaled(corners: Sequence[Sequence[float]], source_size: Sequence[float], target_size: Sequence[float]) -> Quad:
    # Scaling malformed geometry would hide invalid detector output; validate
    # and canonicalize before applying the affine coordinate transform.
    points = validate_quad(corners)
    sw, sh = _size(source_size)
    tw, th = _size(target_size)
    try:
        sx, sy = tw / sw, th / sh
    except (OverflowError, ZeroDivisionError) as exc:
        raise GeometryError("scale ratio must be finite and positive") from exc
    if not math.isfinite(sx) or not math.isfinite(sy) or sx <= 0 or sy <= 0:
        raise GeometryError("scale ratio must be finite and positive")
    out = tuple((x * sx, y * sy) for x, y in points)
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in out):
        raise GeometryError("scaled coordinates must remain finite")
    return out  # type: ignore[return-value]


def to_work_scale(corners: Sequence[Sequence[float]], original_size: Sequence[float], work_size: Sequence[float]) -> Quad:
    return _scaled(corners, original_size, work_size)


def to_original_scale(corners: Sequence[Sequence[float]], work_size: Sequence[float], original_size: Sequence[float]) -> Quad:
    return _scaled(corners, work_size, original_size)


def synthetic_scan(
    photo_quad: Sequence[Sequence[float]] | None = None,
    *,
    quad: Sequence[Sequence[float]] | None = None,
    resolution: Sequence[int] = (320, 240),
    image_size: Sequence[int] | None = None,
    background: Sequence[float] = (232, 232, 228),
    texture: float = 0.0,
    shadow: float = 0.0,
    outer_frame: Sequence[Sequence[float]] | bool | None = None,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, Quad, dict[str, Any]]:
    """Create a deterministic source-derived scan image and truth metadata."""
    if photo_quad is None:
        photo_quad = quad
    elif quad is not None:
        raise GeometryError("provide only one of photo_quad and quad")
    if photo_quad is None:
        raise GeometryError("photo_quad is required")
    if image_size is not None:
        resolution = image_size
    if isinstance(resolution, (str, bytes)) or not isinstance(resolution, Sequence) or len(resolution) != 2:
        raise GeometryError("resolution must be (width, height)")
    width, height = int(resolution[0]), int(resolution[1])
    if width <= 0 or height <= 0 or width != resolution[0] or height != resolution[1]:
        raise ValueError("resolution must contain positive integers")
    if isinstance(background, (str, bytes)) or not isinstance(background, Sequence) or len(background) != 3:
        raise GeometryError("background must contain three channels")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) or not math.isfinite(float(v)) for v in background):
        raise ValueError("background must be finite numeric values")
    background_rgb = np.asarray([float(v) for v in background], dtype=np.float32)
    if any(float(v) < 0 or float(v) > 255 for v in background_rgb):
        raise ValueError("background channels must be in [0, 255]")
    # Canonicalize arbitrary point order first, then reject out-of-image points
    # explicitly instead of letting OpenCV clip them.
    quad = order_quad(photo_quad)
    _check_bounds(quad, width, height)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.broadcast_to(background_rgb, (height, width, 3)).copy()
    # A smooth gradient and deterministic low-amplitude texture form the photo.
    photo = np.stack((45 + 150 * xx / max(width - 1, 1), 35 + 170 * yy / max(height - 1, 1), 90 + 80 * (xx + yy) / max(width + height - 2, 1)), axis=-1)
    if float(texture) < 0 or not math.isfinite(float(texture)):
        raise ValueError("texture must be finite and non-negative")
    if float(texture):
        photo += rng.normal(0.0, min(40.0, 35.0 * float(texture)), size=(height, width, 1))
    mask = np.zeros((height, width), dtype=np.uint8)
    import cv2

    cv2.fillPoly(mask, [np.asarray(quad, dtype=np.int32)], 1)
    image[mask.astype(bool)] = photo[mask.astype(bool)]
    if float(shadow) < 0 or not math.isfinite(float(shadow)):
        raise ValueError("shadow must be finite and non-negative")
    if float(shadow):
        kernel = max(3, int(round(min(width, height) * min(0.25, float(shadow) * 0.08))) * 2 + 1)
        outside = cv2.GaussianBlur(mask, (kernel, kernel), 0).astype(np.float32)
        image -= outside[..., None] * min(80.0, 80.0 * float(shadow)) * 0.35
    if outer_frame is not None and outer_frame is not False:
        frame_quad = quad if outer_frame is True else order_quad(outer_frame)
        _check_bounds(frame_quad, width, height)
        cv2.polylines(image, [np.asarray(frame_quad, dtype=np.int32)], True, (90, 90, 90), max(1, int(round(min(width, height) * 0.01))))
    if float(noise) < 0 or not math.isfinite(float(noise)):
        raise ValueError("noise must be finite and non-negative")
    if float(noise):
        image += rng.normal(0.0, float(noise), size=image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    frame_truth = None if outer_frame is None or outer_frame is False else (quad if outer_frame is True else order_quad(outer_frame))
    if frame_truth is not None:
        _check_bounds(frame_truth, width, height)
    labels: dict[str, Any] = {"photo_quad": quad, "outer_frame": frame_truth, "outer_frame_truth": frame_truth, "seed": int(seed)}
    truth = quad
    return image, truth, labels


# Readable aliases used by downstream providers and tests.
normalize_quad = order_quad
validate_quadrilateral = validate_quad
intersect_lines = line_intersection
area_ratio = quad_area_ratio
normalized_edge_length = min_normalized_edge
original_to_work = to_work_scale
work_to_original = to_original_scale


__all__ = [
    "GeometryError", "order_quad", "normalize_quad", "validate_quad", "validate_quadrilateral", "signed_area",
    "quad_area_ratio", "area_ratio", "min_normalized_edge", "normalized_edge_length", "line_intersection",
    "intersect_lines", "polygon_iou", "normalized_corner_distance", "to_work_scale", "to_original_scale",
    "original_to_work", "work_to_original", "synthetic_scan",
]
