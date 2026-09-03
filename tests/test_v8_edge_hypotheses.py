import cv2
import numpy as np
import pytest

import photocut.algorithms.v8.edge_hypotheses as edge_module
from photocut.algorithms.v8.edge_hypotheses import (
    EdgeHypothesisConfig,
    EdgeQuadHypothesis,
    generate_edge_hypotheses,
)


def _nested_scene():
    image = np.full((300, 400, 3), 250, np.uint8)
    truth = np.asarray(((48, 38), (352, 44), (344, 262), (55, 255)), np.float32)
    cv2.fillConvexPoly(image, truth.astype(np.int32), (240, 242, 244))
    inner = np.asarray(((65, 55), (334, 60), (328, 245), (70, 239)), np.float32)
    cv2.fillConvexPoly(image, inner.astype(np.int32), (55, 105, 175))
    cv2.polylines(image, [truth.astype(np.int32)], True, (222, 225, 228), 2, cv2.LINE_AA)
    return image, truth, inner


def test_edge_hypotheses_preserve_weak_outer_edge_alongside_strong_inner_edge():
    image, truth, inner = _nested_scene()

    result = generate_edge_hypotheses(image, inner)

    assert 1 < len(result.candidates) <= result.config.max_quad_hypotheses
    assert any(np.max(np.linalg.norm(np.asarray(candidate.corners) - truth, axis=1)) <= 5 for candidate in result.candidates)
    assert any(np.max(np.linalg.norm(np.asarray(candidate.corners) - inner, axis=1)) <= 5 for candidate in result.candidates)
    assert all(1 <= len(edge["offsets_px"]) <= result.config.hypotheses_per_edge for edge in result.edge_evidence)


def test_edge_hypotheses_are_deterministic_legal_and_bounded():
    image, _truth, inner = _nested_scene()

    first = generate_edge_hypotheses(image, inner)
    second = generate_edge_hypotheses(image, inner)

    assert first == second
    assert len({candidate.candidate_id for candidate in first.candidates}) == len(first.candidates)
    for candidate in first.candidates:
        assert all(0 <= x <= 399 and 0 <= y <= 299 for x, y in candidate.corners)


def test_edge_evidence_stays_aligned_when_an_earlier_line_fit_fails(monkeypatch):
    image, _truth, inner = _nested_scene()
    original_fit = edge_module._fit_line
    calls = 0

    def fake_profile(_lab, _base, _normal, offsets):
        size = len(offsets)
        return (
            np.arange(size, dtype=np.float64) + 10,
            np.arange(size, dtype=np.float64) + 20,
            np.ones(size, dtype=np.float64),
            np.arange(size, dtype=np.float64) + 30,
        )

    def fail_first_fit_per_edge(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls % 2 == 1:
            raise edge_module.GeometryError("fixture failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(edge_module, "_profile", fake_profile)
    monkeypatch.setattr(edge_module, "_select_positions", lambda *args, **kwargs: [1, 2])
    monkeypatch.setattr(edge_module, "_fit_line", fail_first_fit_per_edge)

    result = generate_edge_hypotheses(image, inner)

    assert len(result.edge_evidence) == 4
    assert all(edge["continuity"] == (12.0,) for edge in result.edge_evidence)
    assert all(edge["white_bed"] == (22.0,) for edge in result.edge_evidence)


def test_vectorized_dedup_matches_max_per_corner_distance_contract():
    accepted = [
        np.asarray(((0, 0), (10, 0), (10, 10), (0, 10)), dtype=np.float64),
        np.asarray(((20, 20), (30, 20), (30, 30), (20, 30)), dtype=np.float64),
    ]
    near = accepted[0] + np.asarray(((0.2, 0.2),) * 4)
    far_on_one_corner = accepted[0].copy()
    far_on_one_corner[2] += (0.6, 0.0)

    assert edge_module._is_near_duplicate(near, accepted, 0.5) is True
    assert edge_module._is_near_duplicate(far_on_one_corner, accepted, 0.5) is False
    assert edge_module._is_near_duplicate(near, [], 0.5) is False


def test_edge_pool_aggregation_preserves_seed_support_and_is_order_independent():
    aggregate_edge_hypotheses = getattr(edge_module, "aggregate_edge_hypotheses", None)
    if aggregate_edge_hypotheses is None:
        pytest.fail("aggregate_edge_hypotheses is missing")
    corners = ((10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0))
    lower = EdgeQuadHypothesis("shared", corners, 10.0, (0.0, 0.0, 0.0, 0.0))
    higher = EdgeQuadHypothesis("shared", corners, 20.0, (0.0, 0.0, 0.0, 0.0))
    observations = (
        ("seed-b", ("background:border_connected",), lower),
        ("seed-a", ("background:min_area_rect",), higher),
    )

    first = aggregate_edge_hypotheses(observations, budget=32)
    second = aggregate_edge_hypotheses(tuple(reversed(observations)), budget=32)

    assert first == second
    assert first == ({
        "candidate_id": "shared",
        "corners": corners,
        "prior_score": 20.0,
        "seed_candidate_ids": ("seed-a", "seed-b"),
        "seed_source_groups": (
            "background:border_connected",
            "background:min_area_rect",
        ),
        "seed_support_count": 2,
        "seed_source_group_count": 2,
    },)


@pytest.mark.parametrize(
    "change",
    [
        {"work_max_edge": 32},
        {"search_band_ratio": 0.2},
        {"hypotheses_per_edge": 9},
        {"max_quad_hypotheses": 1000},
    ],
)
def test_edge_hypothesis_config_rejects_unbounded_values(change):
    with pytest.raises((TypeError, ValueError)):
        EdgeHypothesisConfig(**change)
