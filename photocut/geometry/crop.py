from typing import Sequence, Tuple

import numpy as np


class InsetGeometryError(ValueError):
    """Raised when a safe inset quadrilateral cannot be constructed."""


def _signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _cross_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_x = np.take(first, 0, axis=-1)
    first_y = np.take(first, 1, axis=-1)
    second_x = np.take(second, 0, axis=-1)
    second_y = np.take(second, 1, axis=-1)
    return first_x * second_y - first_y * second_x


def _order_points(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    ordered = points[np.argsort(angles)]
    start = min(
        range(4),
        key=lambda index: (
            ordered[index, 0] + ordered[index, 1],
            ordered[index, 1],
            ordered[index, 0],
        ),
    )
    return np.roll(ordered, -start, axis=0)


def _crosses(points: np.ndarray) -> np.ndarray:
    edges = np.roll(points, -1, axis=0) - points
    return _cross_2d(edges, np.roll(edges, -1, axis=0))


def validate_quadrilateral(
    corners: Sequence[Sequence[float]], image_size: Tuple[int, int]
) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float64)
    width, height = image_size
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise InsetGeometryError("corners must contain four finite points")
    points = _order_points(points)
    if width <= 0 or height <= 0:
        raise InsetGeometryError("image_size must be positive")
    if np.any(points[:, 0] < 0) or np.any(points[:, 0] > width - 1):
        raise InsetGeometryError("corner x coordinate is outside the image")
    if np.any(points[:, 1] < 0) or np.any(points[:, 1] > height - 1):
        raise InsetGeometryError("corner y coordinate is outside the image")
    crosses = _crosses(points)
    if np.any(np.abs(crosses) < 1e-9) or not (
        np.all(crosses > 0) or np.all(crosses < 0)
    ):
        raise InsetGeometryError("corners must form a convex quadrilateral")
    if abs(_signed_area(points)) < 1.0:
        raise InsetGeometryError("quadrilateral area is too small")
    return points


def _line_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> np.ndarray:
    direction_p = p2 - p1
    direction_q = q2 - q1
    denominator = _cross_2d(direction_p, direction_q)
    if abs(float(denominator)) < 1e-9:
        raise InsetGeometryError("adjacent inset edges are parallel")
    factor = _cross_2d(q1 - p1, direction_q) / denominator
    return p1 + factor * direction_p


def inset_quadrilateral(
    corners: Sequence[Sequence[float]],
    distance_px: float,
    image_size: Tuple[int, int],
) -> list[list[float]]:
    if distance_px < 0:
        raise InsetGeometryError("distance_px must be non-negative")
    source = validate_quadrilateral(corners, image_size)
    if distance_px == 0:
        return source.tolist()

    centroid = source.mean(axis=0)
    shifted_edges = []
    normals = []
    for start, end in zip(source, np.roll(source, -1, axis=0)):
        edge = end - start
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        midpoint = (start + end) / 2.0
        if np.dot(centroid - midpoint, normal) < 0:
            normal *= -1
        shifted_edges.append((start + normal * distance_px, end + normal * distance_px))
        normals.append(normal)

    inset = []
    for index in range(4):
        previous = shifted_edges[index - 1]
        current = shifted_edges[index]
        inset.append(_line_intersection(*previous, *current))
    inset_points = np.asarray(inset, dtype=np.float64)

    for edge, normal in zip(source, normals):
        if np.any(np.dot(inset_points - edge, normal) < distance_px - 1e-9):
            raise InsetGeometryError("inset polygon collapses")

    validate_quadrilateral(inset_points, image_size)
    if abs(_signed_area(inset_points)) >= abs(_signed_area(source)):
        raise InsetGeometryError("inset polygon must be smaller than source boundary")
    return np.round(inset_points, 3).tolist()
