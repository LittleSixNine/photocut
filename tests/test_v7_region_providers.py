import time

import cv2
import numpy as np

from photocut.algorithms.v7.features import ImageFeatureContext
from photocut.algorithms.v7.geometry import synthetic_scan
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.providers import (
    BackgroundDifferenceProvider, ContourProvider, WhiteBorderProvider,
)
from photocut.algorithms.v7.types import ProviderStatus


class Token:
    def __init__(self, value=False):
        self.value = value

    def is_cancelled(self):
        return self.value


def _photo():
    image, truth, _ = synthetic_scan(
        ((30, 25), (170, 20), (180, 120), (20, 125)),
        resolution=(200, 150), seed=7, texture=0.2,
    )
    return image, truth


def test_background_provider_returns_original_tl_tr_br_bl_candidate_and_evidence():
    image, truth = _photo()
    result = BackgroundDifferenceProvider().provide(
        ImageFeatureContext(image), V7Parameters(scene_profile="generic_single")
    )
    assert result.status is ProviderStatus.SUCCESS
    candidate = result.candidates[0]
    assert len(candidate["corners"]) == 4
    assert candidate["corners"][0][0] <= candidate["corners"][1][0]
    assert candidate["candidate_id"] == candidate["id"]
    assert candidate["sources"] and candidate["generation_scale"] == 200
    assert candidate["evidence"]["mask"] == "lab_median_mad"
    assert max(np.linalg.norm(np.asarray(a) - np.asarray(b)) for a, b in zip(candidate["corners"], truth)) < 20


def test_scanner_white_adds_only_a_bounded_stable_border_connected_supplement():
    image = np.full((180, 240, 3), 248, dtype=np.uint8)
    truth = np.asarray(((8, 10), (232, 14), (226, 170), (12, 167)), dtype=np.int32)
    cv2.fillConvexPoly(image, truth, (55, 90, 135))
    cv2.rectangle(image, (45, 55), (195, 130), (15, 15, 15), 5)
    context = ImageFeatureContext(image)

    generic = BackgroundDifferenceProvider().provide(
        context, V7Parameters(scene_profile="generic_single")
    )
    scanner = WhiteBorderProvider().provide(
        context, V7Parameters(scene_profile="scanner_white")
    )

    assert all(c["source"] != "background:border_connected" for c in generic.candidates)
    supplements = [c for c in scanner.candidates if c["source"] == "background:border_connected"]
    assert 1 <= len(supplements) <= 3
    assert scanner.diagnostics["border_connected_enabled"] is True
    assert scanner.diagnostics["border_connected_raw_candidates"] >= len(supplements)
    for candidate in supplements:
        evidence = candidate["evidence"]
        assert evidence["supplemental_only"] is True
        assert evidence["reference_support"] >= 1
        assert evidence["threshold_support"] >= 1
    assert min(
        max(np.linalg.norm(np.asarray(a) - np.asarray(b)) for a, b in zip(c["corners"], truth))
        for c in supplements
    ) < 10

    disabled = WhiteBorderProvider().provide(
        context, V7Parameters(scene_profile="generic_single")
    )
    assert disabled.status is ProviderStatus.NO_CANDIDATE
    assert disabled.candidates == ()


def test_contour_provider_is_deterministic_and_uses_cached_features():
    image, _ = _photo()
    context = ImageFeatureContext(image)
    params = V7Parameters()
    first = ContourProvider().provide(context, params)
    second = ContourProvider().provide(context, params)
    assert first.status is second.status
    assert first.candidates == second.candidates
    assert len({c["id"] for c in first.candidates}) == len(first.candidates)
    assert first.work_consumed == second.work_consumed
    assert first.diagnostics == second.diagnostics
    assert first.status is ProviderStatus.SUCCESS
    assert context.build_counts["edges"] == 1
    assert all(c["evidence"]["retrieval"] == "ccomp" for c in first.candidates)


def test_provider_statuses_discard_partial_candidates_on_cancel_timeout_and_budget():
    image, _ = _photo()
    cancelled = Token(True)
    result = BackgroundDifferenceProvider().provide(ImageFeatureContext(image), V7Parameters(), cancelled)
    assert result.status is ProviderStatus.CANCELLED
    assert result.candidates == ()

    expired = BackgroundDifferenceProvider().provide(ImageFeatureContext(image), V7Parameters(), deadline=time.monotonic() - 1)
    assert expired.status is ProviderStatus.TIMEOUT
    assert expired.candidates == ()

    tiny = V7Parameters().replace(provider_work_limits={"background": 1})
    exhausted = BackgroundDifferenceProvider().provide(ImageFeatureContext(image), tiny)
    assert exhausted.status is ProviderStatus.BUDGET_EXHAUSTED
    assert exhausted.candidates == ()
    assert exhausted.work_consumed == exhausted.work_limit == 1
    contour_tiny = V7Parameters().replace(provider_work_limits={"contour": 2})
    contour_result = ContourProvider().provide(ImageFeatureContext(image), contour_tiny)
    assert contour_result.status is ProviderStatus.BUDGET_EXHAUSTED
    assert contour_result.candidates == ()


def test_contour_degenerate_four_point_approximation_uses_bounded_rect_fallback(monkeypatch):
    image, _ = _photo()
    original = cv2.approxPolyDP

    def degenerate(*args, **kwargs):
        return np.asarray([[(0, 0)], [(10, 10)], [(0, 10)], [(10, 0)]], dtype=np.int32)

    monkeypatch.setattr(cv2, "approxPolyDP", degenerate)
    result = ContourProvider().provide(ImageFeatureContext(image), V7Parameters())
    assert result.status is ProviderStatus.SUCCESS
    assert any(c["source"] == "contour:convex_hull_rect" for c in result.candidates)
    monkeypatch.setattr(cv2, "approxPolyDP", original)


def test_background_no_photo_is_explicit_no_candidate_and_multiple_regions_have_ambiguity_evidence():
    blank = np.full((150, 200, 3), 245, dtype=np.uint8)
    no_photo = BackgroundDifferenceProvider().provide(ImageFeatureContext(blank), V7Parameters())
    assert no_photo.status is ProviderStatus.NO_CANDIDATE
    assert no_photo.candidates == ()

    image = blank.copy()
    cv2.rectangle(image, (15, 20), (85, 120), (45, 60, 80), -1)
    cv2.rectangle(image, (115, 25), (185, 125), (55, 70, 95), -1)
    multiple = BackgroundDifferenceProvider().provide(ImageFeatureContext(image), V7Parameters())
    assert multiple.status is ProviderStatus.SUCCESS
    assert len(multiple.candidates) >= 2
    assert multiple.diagnostics["candidate_count"] >= 2


def test_background_uses_fixed_lab_mask_variants_on_yellow_and_gray_scanners():
    truth = ((30.0, 25.0), (170.0, 25.0), (170.0, 125.0), (30.0, 125.0))
    for background in ((230, 220, 170), (150, 150, 150)):
        image = np.full((150, 200, 3), background, dtype=np.uint8)
        cv2.fillConvexPoly(image, np.asarray(truth, dtype=np.int32), (45, 65, 85))
        result = BackgroundDifferenceProvider().provide(
            ImageFeatureContext(image), V7Parameters(scene_profile="generic_single")
        )
        assert result.status is ProviderStatus.SUCCESS
        variants = {c["evidence"]["mask_variant"] for c in result.candidates}
        assert {"lab_max", "l_channel", "chroma"}.issubset(variants)
        assert min(max(np.linalg.norm(np.asarray(a) - np.asarray(b)) for a, b in zip(c["corners"], truth)) for c in result.candidates) < 20


def test_background_rejects_out_of_bounds_quad_approximation(monkeypatch):
    image, _ = _photo()
    original = cv2.approxPolyDP

    def outside(*args, **kwargs):
        return np.asarray([[(-100, -100)], [(500, -100)], [(500, 500)], [(-100, 500)]], dtype=np.int32)

    monkeypatch.setattr(cv2, "approxPolyDP", outside)
    result = BackgroundDifferenceProvider().provide(
        ImageFeatureContext(image), V7Parameters(scene_profile="generic_single")
    )
    monkeypatch.setattr(cv2, "approxPolyDP", original)
    assert result.status is ProviderStatus.NO_CANDIDATE
    assert result.candidates == ()
