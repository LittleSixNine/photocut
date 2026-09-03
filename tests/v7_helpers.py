"""Small deterministic helpers shared by v7 geometry tests."""
from __future__ import annotations

from typing import Sequence


def quad(x: float = 10.0, y: float = 10.0, width: float = 80.0, height: float = 60.0):
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))


def scale_quad(corners: Sequence[Sequence[float]], scale: float):
    if not isinstance(scale, (int, float)) or isinstance(scale, bool) or scale <= 0:
        raise ValueError("scale must be positive")
    return tuple((float(x) * scale, float(y) * scale) for x, y in corners)


def translate_quad(corners: Sequence[Sequence[float]], dx: float, dy: float):
    return tuple((float(x) + dx, float(y) + dy) for x, y in corners)


def assert_quad_close(actual, expected, tolerance: float = 1e-6):
    for (ax, ay), (ex, ey) in zip(actual, expected):
        assert abs(ax - ex) <= tolerance
        assert abs(ay - ey) <= tolerance
