import math

import numpy as np
import pytest

from photocut.algorithms.v7.geometry import (
    GeometryError,
    line_intersection,
    min_normalized_edge,
    normalized_corner_distance,
    order_quad,
    polygon_iou,
    quad_area_ratio,
    signed_area,
    synthetic_scan,
    to_original_scale,
    to_work_scale,
    validate_quad,
)


def test_order_quad_returns_tl_tr_br_bl_for_rotated_and_perspective_input():
    expected = ((18.0, 28.0), (172.0, 8.0), (190.0, 134.0), (4.0, 158.0))
    actual = order_quad([expected[2], expected[0], expected[3], expected[1]])
    assert actual == expected
    assert signed_area(actual) > 0


@pytest.mark.parametrize(
    "quad",
    [
        ((0, 0), (1, 1), (0, 1), (1, 0)),
        ((0, 0), (2, 0), (1, 0.5), (0, 2)),
        ((0, 0), (1, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (float("nan"), 1), (0, 1)),
        ((0, 0), (1, 0), (float("inf"), 1), (0, 1)),
    ],
)
def test_order_quad_rejects_duplicate_nonfinite_self_intersecting_or_concave(quad):
    with pytest.raises((GeometryError, TypeError, ValueError)):
        validate_quad(quad)


def test_scale_safe_quality_iou_and_corner_distance():
    q = ((10.0, 10.0), (90.0, 8.0), (92.0, 70.0), (8.0, 72.0))
    q2 = tuple((x + 2.0, y - 1.0) for x, y in q)
    assert quad_area_ratio(q, (100, 100)) == pytest.approx(0.5084)
    assert min_normalized_edge(q, (100, 100)) > 0.4
    assert polygon_iou(q, q) == pytest.approx(1.0)
    assert 0 < polygon_iou(q, q2) < 1
    assert normalized_corner_distance(q, q2, (100, 100)) == pytest.approx(math.sqrt(5) / math.sqrt(2 * 100**2))


def test_near_parallel_lines_are_rejected():
    with pytest.raises(GeometryError):
        line_intersection((0, 0), (1000, 1), (0, 1), (1000, 2.0000001))


def test_work_original_round_trip_is_immutable_and_scale_safe():
    original = ((100.25, 40.5), (900.5, 20.25), (930.75, 700.0), (80.0, 730.125))
    work = to_work_scale(original, (1000, 800), (500, 400))
    back = to_original_scale(work, (500, 400), (1000, 800))
    assert isinstance(work, tuple) and all(isinstance(p, tuple) for p in work)
    assert np.max(np.abs(np.asarray(back) - original)) <= 1e-9


def test_geometry_is_translation_invariant_and_scale_rejects_invalid_quads():
    base = ((0.0, 0.0), (100.0, 2.0), (98.0, 80.0), (-2.0, 78.0))
    translated = tuple((x + 1e9, y + 1e9) for x, y in base)
    assert order_quad(translated) == tuple((x + 1e9, y + 1e9) for x, y in order_quad(base))
    with pytest.raises((GeometryError, TypeError, ValueError)):
        to_work_scale(((0, 0), (10, 10), (0, 10), (10, 0)), (10, 10), (5, 5))


def test_extreme_translation_and_image_bounds_are_safe():
    base = ((0.0, 0.0), (100.0, 2.0), (98.0, 80.0), (-2.0, 78.0))
    translated = tuple((x + 1e12, y + 1e12) for x, y in base)
    assert signed_area(order_quad(translated)) == pytest.approx(signed_area(order_quad(base)))
    with pytest.raises(GeometryError):
        validate_quad(((0, 0), (101, 0), (101, 50), (0, 50)), (100, 100))
    with pytest.raises(GeometryError):
        validate_quad(((-1, 0), (50, 0), (50, 50), (0, 50)), (100, 100))


def test_polygon_iou_rejects_concave_non_quad_polygons():
    concave = ((0, 0), (4, 0), (2, 1), (4, 4), (0, 4))
    with pytest.raises(GeometryError):
        polygon_iou(concave, concave)


def test_scale_and_intersection_reject_nonfinite_controls():
    q = ((1.0, 1.0), (9.0, 1.0), (9.0, 9.0), (1.0, 9.0))
    with pytest.raises(GeometryError):
        to_work_scale(q, (1e-308, 1e-308), (1e308, 1e308))
    with pytest.raises(GeometryError):
        line_intersection((0, 0), (1, 0), (0, 1), (1, 1), tolerance=-1)
    with pytest.raises(GeometryError):
        line_intersection((0, 0), (1, 0), (0, 1), (1, 1), tolerance=float("nan"))


def test_synthetic_scan_rejects_out_of_bounds_quads():
    with pytest.raises(GeometryError):
        synthetic_scan(((0, 0), (210, 0), (210, 100), (0, 100)), resolution=(200, 100))
    with pytest.raises(GeometryError):
        synthetic_scan(((0, 0), (100, 0), (100, 100), (0, 100)), resolution=(200, 100), outer_frame=((0, 0), (201, 0), (201, 99), (0, 99)))


def test_synthetic_scan_canonicalizes_arbitrary_point_order():
    ordered = ((20, 18), (180, 10), (188, 118), (12, 126))
    shuffled = (ordered[2], ordered[0], ordered[3], ordered[1])
    image, truth, labels = synthetic_scan(shuffled, resolution=(200, 140), seed=3)
    assert truth == order_quad(ordered)
    assert labels["photo_quad"] == truth
    assert image.shape == (140, 200, 3)


def test_synthetic_scan_is_deterministic_and_returns_truth_labels():
    quad = ((20, 18), (180, 10), (188, 118), (12, 126))
    image_a, truth_a, labels_a = synthetic_scan(quad, resolution=(200, 140), seed=7, noise=3, texture=0.2, shadow=0.3)
    image_b, truth_b, labels_b = synthetic_scan(quad, resolution=(200, 140), seed=7, noise=3, texture=0.2, shadow=0.3)
    assert image_a.shape == (140, 200, 3)
    assert image_a.dtype == np.uint8
    assert np.array_equal(image_a, image_b)
    assert truth_a == truth_b == order_quad(quad)
    assert labels_a == labels_b
    assert labels_a["photo_quad"] == truth_a
