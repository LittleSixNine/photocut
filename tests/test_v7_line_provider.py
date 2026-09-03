import time

import cv2
import numpy as np

from photocut.algorithms.v7.features import ImageFeatureContext
from photocut.algorithms.v7.geometry import synthetic_scan
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.providers import BackgroundDifferenceProvider, ContourProvider, LineProvider
from photocut.algorithms.v7.types import ProviderStatus


class Token:
    def __init__(self, value=False):
        self.value = value

    def is_cancelled(self):
        return self.value


def _scan(quad, *, texture=0.0):
    image, truth, _ = synthetic_scan(quad, resolution=(320, 240), seed=13, texture=texture, noise=1)
    return image, truth


def test_line_provider_finds_rotated_scan_and_is_deterministic():
    image, truth = _scan(((36, 57), (258, 20), (280, 182), (16, 210)), texture=0.2)
    params = V7Parameters().replace(provider_timeout_ms={"lines": 2000})
    first = LineProvider().provide(ImageFeatureContext(image), params)
    second = LineProvider().provide(ImageFeatureContext(image), params)
    assert first.status is ProviderStatus.SUCCESS
    assert first.candidates == second.candidates
    assert len(first.candidates) <= 64
    candidate = min(first.candidates, key=lambda item: max(np.linalg.norm(np.asarray(a) - np.asarray(b)) for a, b in zip(item["corners"], truth)))
    assert len(candidate["corners"]) == 4
    assert max(np.linalg.norm(np.asarray(a) - np.asarray(b)) for a, b in zip(candidate["corners"], truth)) < 35
    assert first.diagnostics["raw_segments"] <= 256
    assert first.diagnostics["merged_lines"] <= 64
    assert first.diagnostics["families"] <= 36
    assert first.diagnostics["pairs"] <= 256


def test_line_provider_handles_trapezoid_and_grid_without_combination_explosion():
    image, _ = _scan(((26, 30), (292, 46), (264, 208), (48, 188)), texture=0.4)
    cv2.line(image, (0, 80), (319, 80), (220, 220, 220), 1)
    cv2.line(image, (110, 0), (110, 239), (220, 220, 220), 1)
    result = LineProvider().provide(ImageFeatureContext(image), V7Parameters().replace(provider_timeout_ms={"lines": 2000}))
    assert result.status in (ProviderStatus.SUCCESS, ProviderStatus.NO_CANDIDATE)
    assert len(result.candidates) <= 64
    assert result.diagnostics["pairs"] <= 256


def test_line_provider_cancellation_timeout_and_budget_discard_partial_candidates():
    image, _ = _scan(((30, 25), (280, 25), (270, 210), (40, 220)), texture=0.2)
    context = ImageFeatureContext(image)
    cancelled = LineProvider().provide(context, V7Parameters(), Token(True))
    assert cancelled.status is ProviderStatus.CANCELLED
    assert cancelled.candidates == ()
    timeout = LineProvider().provide(ImageFeatureContext(image), V7Parameters(), deadline=time.monotonic() - 1)
    assert timeout.status is ProviderStatus.TIMEOUT
    assert timeout.candidates == ()
    tiny = V7Parameters().replace(provider_work_limits={"lines": 1}, provider_timeout_ms={"lines": 2000})
    exhausted = LineProvider().provide(ImageFeatureContext(image), tiny)
    assert exhausted.status is ProviderStatus.BUDGET_EXHAUSTED
    assert exhausted.candidates == ()
    assert exhausted.work_consumed == exhausted.work_limit == 1


def test_all_provider_timeout_mapping_discards_every_provider_prefix():
    image, _ = _scan(((30, 25), (280, 25), (270, 210), (40, 220)))
    for provider in (BackgroundDifferenceProvider(), ContourProvider(), LineProvider()):
        result = provider.provide(ImageFeatureContext(image), V7Parameters(), deadline=time.monotonic() - 1)
        assert result.status is ProviderStatus.TIMEOUT
        assert result.candidates == ()
        assert result.timeout_code == "provider_deadline"
