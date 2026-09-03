import cv2
import time
import numpy as np
import pytest

from photocut.algorithms.v7.features import ImageFeatureContext
from photocut.algorithms.v7.geometry import synthetic_scan
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.refinement import RefinementResult, refine_candidate, refine_quad


def _image(quad, *, noise=0, shadow=False, internal=False):
    image, truth, _ = synthetic_scan(
        quad, resolution=(320, 240), seed=19, noise=noise, shadow=0.25 if shadow else 0.0
    )
    if internal:
        cv2.line(image, (80, 0), (80, 239), (255, 255, 255), 2)
        cv2.line(image, (150, 0), (150, 239), (255, 255, 255), 2)
    return image, truth


def test_refinement_returns_proposed_and_adopted_corners_for_noisy_edges():
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)), noise=3)
    result = refine_quad(ImageFeatureContext(image), truth, params=V7Parameters())
    assert isinstance(result, RefinementResult)
    assert result.proposed_corners is not None
    assert result.adopted_corners == result.proposed_corners or result.adopted_corners == result.original_corners
    assert len(result.edge_evidence) == 4
    assert all("support" in edge for edge in result.edge_evidence)


def test_broken_edges_and_shadows_never_make_geometry_worse():
    image, truth = _image(((25, 34), (294, 45), (271, 207), (46, 192)), noise=5, shadow=True)
    result = refine_quad(ImageFeatureContext(image), truth, params=V7Parameters())
    assert result.adopted_corners in (result.original_corners, result.proposed_corners)
    assert result.max_normalized_shift <= V7Parameters().max_refinement_shift + 1e-12


def test_internal_parallel_lines_do_not_replace_candidate_edges():
    image, truth = _image(((38, 28), (278, 35), (265, 205), (42, 193)), internal=True)
    result = refine_quad(ImageFeatureContext(image), truth, params=V7Parameters())
    assert result.adopted_corners in (result.original_corners, result.proposed_corners)
    assert "unsafe" not in " ".join(result.risks)


def test_insufficient_support_explicitly_rolls_back():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    original = ((30.0, 30.0), (290.0, 30.0), (290.0, 210.0), (30.0, 210.0))
    result = refine_quad(ImageFeatureContext(image), original, params=V7Parameters())
    assert result.adopted_corners == result.original_corners
    assert "insufficient_support" in result.risks


def test_near_parallel_fits_roll_back_with_structured_risk(monkeypatch):
    image, _ = _image(((40, 40), (280, 40), (280, 200), (40, 200)))
    def parallel_fit(points, params, diagonal):
        return (1.0, 0.0, 100.0, 100.0), {"support": float(len(points)), "residual": 0.0}, None
    monkeypatch.setattr("photocut.algorithms.v7.refinement._fit_line", parallel_fit)
    original = ((40.0, 40.0), (280.0, 40.0), (280.0, 200.0), (40.0, 200.0))
    result = refine_quad(ImageFeatureContext(image), original, params=V7Parameters())
    assert result.adopted_corners == result.original_corners
    assert "near_parallel_intersection" in result.risks


def test_excessive_shift_rolls_back():
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)))
    result = refine_quad(ImageFeatureContext(image), truth, params=V7Parameters(max_refinement_shift=1e-6))
    assert result.adopted_corners == result.original_corners
    assert "excessive_shift" in result.risks


def test_refinement_is_deterministic_and_audit_has_no_truth_metrics():
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)), noise=2)
    context = ImageFeatureContext(image)
    a = refine_quad(context, truth, params=V7Parameters())
    b = refine_quad(context, truth, params=V7Parameters())
    assert a == b
    audit = a.to_audit("c1", sources=("contour",))
    assert audit.proposed_refined_corners == a.proposed_corners
    assert audit.adopted_refined_corners == a.adopted_corners
    assert not any("truth" in str(key).lower() or "iou" in str(key).lower() for key in audit.pre_truncation_risk_evidence)


@pytest.mark.parametrize("token", [
    type("PropertyToken", (), {"is_cancelled": False})(),
    type("CancelledPropertyToken", (), {"cancelled": False})(),
    type("EventToken", (), {"is_set": lambda self: False})(),
])
def test_false_property_or_event_tokens_do_not_cancel_refinement(token):
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)))
    result = refine_quad(ImageFeatureContext(image), truth, params=V7Parameters(), cancellation_token=token)
    assert "refinement_cancelled" not in result.risks


@pytest.mark.parametrize("deadline", [
    time.monotonic() + 300.0,
    lambda: False,
    type("Expiry", (), {"is_expired": False})(),
])
def test_future_or_false_deadlines_do_not_timeout(deadline):
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)))
    result = refine_quad(ImageFeatureContext(image), truth, params=V7Parameters(), deadline=deadline)
    assert "refinement_timeout" not in result.risks


def test_refine_candidate_accepts_numpy_corners_in_reverse_argument_order():
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)))
    context = ImageFeatureContext(image)
    result = refine_candidate(np.asarray(truth, dtype=np.float32), context, V7Parameters())
    assert result.original_corners == tuple(tuple(float(v) for v in point) for point in truth)


def test_refinement_features_and_global_gradient_stats_are_cached_once():
    image, truth = _image(((34, 46), (278, 28), (286, 194), (22, 211)))
    context = ImageFeatureContext(image)
    refine_quad(context, truth, params=V7Parameters())
    refine_quad(context, truth, params=V7Parameters())
    assert context.build_counts["refinement"] == 1
    assert context.build_counts["gradient"] == 1


def test_max_normalized_edge_gate_rejects_overlong_candidate():
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    original = ((0.0, 0.0), (319.0, 0.0), (319.0, 239.0), (0.0, 239.0))
    with pytest.raises(ValueError, match="too-long normalized edge"):
        refine_quad(ImageFeatureContext(image), original, params=V7Parameters(max_edge_ratio=0.5))
