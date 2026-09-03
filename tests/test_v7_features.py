import numpy as np
import pytest

from photocut.algorithms.v7.features import ImageFeatureContext, FeatureCancelled


class Token:
    cancelled = False


def test_cache_builds_each_expensive_feature_once(monkeypatch):
    image = np.zeros((20, 30, 3), np.uint8)
    calls = {name: 0 for name in ("resize", "cvt", "sobel", "canny")}
    real_resize, real_cvt = __import__("cv2").resize, __import__("cv2").cvtColor
    import cv2
    monkeypatch.setattr(cv2, "resize", lambda *a, **k: (calls.__setitem__("resize", calls["resize"] + 1) or real_resize(*a, **k)))
    monkeypatch.setattr(cv2, "cvtColor", lambda *a, **k: (calls.__setitem__("cvt", calls["cvt"] + 1) or real_cvt(*a, **k)))
    real_sobel, real_canny = cv2.Sobel, cv2.Canny
    monkeypatch.setattr(cv2, "Sobel", lambda *a, **k: (calls.__setitem__("sobel", calls["sobel"] + 1) or real_sobel(*a, **k)))
    monkeypatch.setattr(cv2, "Canny", lambda *a, **k: (calls.__setitem__("canny", calls["canny"] + 1) or real_canny(*a, **k)))

    context = ImageFeatureContext(image, Token())
    context.bgr(10); context.bgr(10)
    context.gray(10); context.gray(10)
    context.lab(10); context.lab(10)
    context.gradient(10); context.gradient(10)
    context.edges(10); context.edges(10)
    context.bgr(20)
    assert calls["resize"] == 2
    assert calls["cvt"] == 2  # gray and Lab, each once
    assert calls["sobel"] == 2
    assert calls["canny"] == 1
    assert context.build_counts["gray"] == 1
    assert context.build_counts["edges"] == 1
    assert context.pixels_visited > 0


def test_views_are_readonly_and_close_releases_cache():
    context = ImageFeatureContext(np.zeros((4, 5, 3), np.uint8), Token())
    arr = context.gray()
    assert arr.flags.writeable is False
    context.close()
    with pytest.raises(RuntimeError, match="closed"):
        context.gray()


def test_original_resolution_none_and_explicit_edge_share_cache():
    context = ImageFeatureContext(np.zeros((4, 5, 3), np.uint8), Token())
    context.bgr()
    context.bgr(5)
    assert context.build_counts["bgr"] == 1


def test_target_edge_one_is_exact_not_fractional():
    context = ImageFeatureContext(np.zeros((4, 5, 3), np.uint8), Token())
    assert context.bgr(target_edge=1).shape[:2] == (1, 1)


def test_cancellation_is_checked_before_work():
    token = Token(); token.cancelled = True
    context = ImageFeatureContext(np.zeros((4, 5, 3), np.uint8), token)
    with pytest.raises(FeatureCancelled):
        context.bgr()
