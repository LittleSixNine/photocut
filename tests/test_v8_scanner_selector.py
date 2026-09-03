import math
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import photocut.algorithms.v8.scanner_selector as selector_module
from photocut.algorithms.v8.scanner_selector import (
    ScannerExteriorConfig,
    evaluate_scanner_exterior,
    rank_scanner_candidates,
    select_scanner_candidate,
)


def _nested_white_bed_scene():
    image = np.full((300, 400, 3), 250, np.uint8)
    truth = np.asarray(((40, 30), (360, 30), (360, 270), (40, 270)), np.float32)
    cv2.fillConvexPoly(image, truth.astype(np.int32), (226, 229, 232))
    inner = np.asarray(((62, 52), (338, 52), (338, 248), (62, 248)), np.float32)
    cv2.fillConvexPoly(image, inner.astype(np.int32), (45, 95, 165))
    cv2.polylines(image, [truth.astype(np.int32)], True, (214, 218, 222), 2, cv2.LINE_AA)
    return image, truth, inner


def test_outer_physical_edge_scores_above_stronger_inner_print_edge():
    image, truth, inner = _nested_white_bed_scene()

    outer_evidence = evaluate_scanner_exterior(image, truth)
    inner_evidence = evaluate_scanner_exterior(image, inner)

    assert outer_evidence.valid_side_count == 4
    assert inner_evidence.valid_side_count == 4
    assert outer_evidence.score > inner_evidence.score + 0.10
    assert outer_evidence.side_stability_mean > inner_evidence.side_stability_mean


def test_joint_ranker_rejects_candidate_with_one_internal_side():
    image, truth, _inner = _nested_white_bed_scene()
    one_side_inside = np.asarray(((62, 54), (338, 54), (360, 270), (40, 270)), np.float32)
    candidates = (
        {"candidate_id": "one-side-inside", "corners": one_side_inside, "prior_score": 0.5},
        {"candidate_id": "physical-edge", "corners": truth, "prior_score": 0.5},
    )

    ranked = rank_scanner_candidates(image, candidates)

    assert [item.candidate_id for item in ranked] == ["physical-edge", "one-side-inside"]
    assert ranked[0].evidence.side_score_min > ranked[1].evidence.side_score_min


def test_joint_ranker_prepares_scanner_bed_reference_once_for_all_candidates():
    image, truth, inner = _nested_white_bed_scene()
    candidates = (
        {"candidate_id": "outer", "corners": truth, "prior_score": 0.5},
        {"candidate_id": "inner", "corners": inner, "prior_score": 0.5},
    )

    with patch(
        "photocut.algorithms.v8.scanner_selector._scanner_bed_reference",
        wraps=selector_module._scanner_bed_reference,
    ) as reference:
        ranked = rank_scanner_candidates(image, candidates)

    assert len(ranked) == 2
    assert reference.call_count == 1


def test_selector_uses_nearest_edge_when_base_choice_strongly_disagrees_with_v7():
    image, truth, inner = _nested_white_bed_scene()
    too_large = np.asarray(((18, 14), (382, 14), (382, 286), (18, 286)), np.float32)
    edge_candidates = (
        {"candidate_id": "too-large", "corners": too_large, "prior_score": 1.0},
        {"candidate_id": "physical-edge", "corners": truth, "prior_score": 0.0},
        {"candidate_id": "inner", "corners": inner, "prior_score": 0.4},
    )

    selected = select_scanner_candidate(
        image,
        edge_candidates,
        v7_candidate={"candidate_id": "v7", "corners": truth},
    )

    assert selected.base_edge_candidate_id == "too-large"
    assert selected.selected.candidate_id == "physical-edge"
    assert selected.reason == "v7_edge_agreement_rescue"


def test_selector_prefers_multi_seed_physical_edge_over_single_seed_larger_outlier():
    image, truth, _inner = _nested_white_bed_scene()
    too_large = np.asarray(((18, 14), (382, 14), (382, 286), (18, 286)), np.float32)
    edge_candidates = (
        {
            "candidate_id": "too-large-single-seed",
            "corners": too_large,
            "prior_score": 1.0,
            "seed_support_count": 1,
            "seed_source_group_count": 1,
        },
        {
            "candidate_id": "physical-multi-seed",
            "corners": truth,
            "prior_score": 0.0,
            "seed_support_count": 3,
            "seed_source_group_count": 2,
        },
    )

    selected = select_scanner_candidate(image, edge_candidates)

    assert selected.base_edge_candidate_id == "too-large-single-seed"
    assert selected.selected.candidate_id == "physical-multi-seed"
    assert selected.reason == "edge_seed_support_rescue"


def test_selector_uses_v7_when_v7_and_mask_agree_but_edge_is_internal():
    image, truth, inner = _nested_white_bed_scene()
    mask = truth + np.asarray(((1.0, 0.0),) * 4, np.float32)

    selected = select_scanner_candidate(
        image,
        ({"candidate_id": "inner", "corners": inner, "prior_score": 1.0},),
        v7_candidate={"candidate_id": "v7", "corners": truth},
        mask_candidate={"candidate_id": "mask", "corners": mask},
    )

    assert selected.selected.candidate_id == "v7"
    assert selected.reason == "v7_mask_consensus_rescue"


def test_selector_uses_mask_when_low_prior_edge_agrees_and_v7_is_far():
    image, truth, inner = _nested_white_bed_scene()
    near_outer = np.asarray(((30, 20), (370, 20), (370, 280), (30, 280)), np.float32)
    v7_far = np.asarray(((125, 105), (275, 105), (275, 195), (125, 195)), np.float32)
    edge_candidates = (
        {"candidate_id": "near-outer", "corners": near_outer, "prior_score": 0.0},
        {"candidate_id": "inner", "corners": inner, "prior_score": 1.0},
    )

    selected = select_scanner_candidate(
        image,
        edge_candidates,
        v7_candidate={"candidate_id": "v7", "corners": v7_far},
        mask_candidate={"candidate_id": "mask", "corners": truth},
    )

    assert selected.base_edge_candidate_id == "near-outer"
    assert selected.selected.candidate_id == "mask"
    assert selected.reason == "edge_mask_consensus_rescue"


def test_border_touching_photo_uses_available_sides_without_nan_or_rejection():
    image = np.full((240, 320, 3), 250, np.uint8)
    truth = np.asarray(((0, 0), (319, 0), (300, 220), (18, 220)), np.float32)
    cv2.fillConvexPoly(image, truth.astype(np.int32), (80, 120, 180))

    first = evaluate_scanner_exterior(image, truth)
    second = evaluate_scanner_exterior(image, truth)

    assert first == second
    assert 2 <= first.valid_side_count <= 4
    assert math.isfinite(first.score)
    assert 0.0 <= first.score <= 1.0


def test_ranker_is_deterministic_and_does_not_mutate_candidates():
    image, truth, inner = _nested_white_bed_scene()
    candidates = [
        {"candidate_id": "inner", "corners": inner.tolist(), "prior_score": 0.8},
        {"candidate_id": "outer", "corners": truth.tolist(), "prior_score": 0.4},
    ]
    original = [dict(item) for item in candidates]

    first = rank_scanner_candidates(image, candidates)
    second = rank_scanner_candidates(image, tuple(reversed(candidates)))

    assert [(item.candidate_id, item.score) for item in first] == [
        (item.candidate_id, item.score) for item in second
    ]
    assert candidates == original


def test_selector_reports_next_geometrically_distinct_edge_competitor():
    image, truth, inner = _nested_white_bed_scene()
    near_truth = truth + np.asarray(((1.0, 0.0),) * 4, np.float32)
    candidates = (
        {"candidate_id": "outer", "corners": truth, "prior_score": 1.0},
        {"candidate_id": "outer-near", "corners": near_truth, "prior_score": 0.9},
        {"candidate_id": "inner", "corners": inner, "prior_score": 0.8},
    )

    selected = select_scanner_candidate(image, candidates)

    assert getattr(selected, "competing_edge_candidate_id", None) == "inner"
    assert getattr(selected, "competing_edge_distance", None) > 0.03
    assert math.isfinite(getattr(selected, "edge_score_margin", math.nan))


def test_selector_aggregates_seed_support_within_selected_geometry_cluster():
    image, truth, _inner = _nested_white_bed_scene()
    near_truth = truth + np.asarray(((1.0, 0.0),) * 4, np.float32)
    candidates = (
        {
            "candidate_id": "outer-a",
            "corners": truth,
            "prior_score": 1.0,
            "seed_candidate_ids": ("seed-a",),
            "seed_source_groups": ("background:min_area_rect",),
            "seed_support_count": 1,
            "seed_source_group_count": 1,
        },
        {
            "candidate_id": "outer-b",
            "corners": near_truth,
            "prior_score": 0.9,
            "seed_candidate_ids": ("seed-b",),
            "seed_source_groups": ("background:border_connected",),
            "seed_support_count": 1,
            "seed_source_group_count": 1,
        },
    )

    selected = select_scanner_candidate(image, candidates)

    assert getattr(selected, "selected_edge_cluster_size", None) == 2
    assert getattr(selected, "selected_edge_cluster_seed_support_count", None) == 2
    assert getattr(selected, "selected_edge_cluster_source_group_count", None) == 2


def test_selector_prefers_supported_geometry_cluster_over_single_seed_larger_outlier():
    image, truth, _inner = _nested_white_bed_scene()
    too_large = np.asarray(((18, 14), (382, 14), (382, 286), (18, 286)), np.float32)
    near_truth = truth + np.asarray(((1.0, 0.0),) * 4, np.float32)
    candidates = (
        {
            "candidate_id": "too-large",
            "corners": too_large,
            "prior_score": 1.0,
            "seed_candidate_ids": ("seed-outlier",),
            "seed_source_groups": ("background:frame_completion",),
        },
        {
            "candidate_id": "physical-a",
            "corners": truth,
            "prior_score": 0.1,
            "seed_candidate_ids": ("seed-a",),
            "seed_source_groups": ("background:min_area_rect",),
        },
        {
            "candidate_id": "physical-b",
            "corners": near_truth,
            "prior_score": 0.0,
            "seed_candidate_ids": ("seed-b",),
            "seed_source_groups": ("background:border_connected",),
        },
    )

    selected = select_scanner_candidate(image, candidates)

    assert selected.base_edge_candidate_id == "too-large"
    assert selected.selected.candidate_id in {"physical-a", "physical-b"}
    assert selected.reason == "edge_cluster_support_rescue"


@pytest.mark.parametrize(
    "change",
    (
        {"work_max_edge": 32},
        {"samples_per_side": 8},
        {"outward_distance_ratio": 0.5},
        {"distance_samples": 100},
    ),
)
def test_scanner_exterior_config_rejects_unbounded_values(change):
    with pytest.raises((TypeError, ValueError)):
        ScannerExteriorConfig(**change)
