import numpy as np

from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.types import DetectionStatus, ProviderResult, ProviderStatus


def _image():
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    image[10:90, 20:120] = 200
    return image


class Provider:
    name = "synthetic"

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def provide(self, context, params, cancellation_token=None, deadline=None):
        self.calls += 1
        return self.result


def candidate(cid="photo", score=0.9):
    return {"candidate_id": cid, "id": cid,
            "corners": ((20, 10), (119, 10), (119, 89), (20, 89)),
            "sources": ("synthetic",), "source": "synthetic", "score": score,
            "evidence": {"photo_content": True}}


def test_detector_returns_recommended_and_closes_context(monkeypatch):
    provider = Provider(ProviderResult("synthetic", (candidate(),), ProviderStatus.SUCCESS))
    result = __import__("photocut.algorithms.v7.detector", fromlist=["detect_corners_v7"]).detect_corners_v7(
        _image(), providers=(provider,), params=V7Parameters(), request_id="r1"
    )
    assert result.status in {DetectionStatus.V7_RECOMMENDED, DetectionStatus.V7_LOW_CONFIDENCE}
    assert np.allclose(result.corners, candidate()["corners"], atol=2.0)
    assert result.identity.request_id == "r1"
    assert result.candidate_audit
    assert provider.calls == 1


def test_zero_candidates_use_injected_v52_without_mutation():
    seen = []
    original = ([[9, 8], [7, 6], [5, 4], [3, 2]], [0.1, 0.2, 0.3, 0.4], [{"legacy": 1}])

    def fallback(image, **kwargs):
        seen.append((image, kwargs))
        return original

    provider = Provider(ProviderResult("synthetic", (), ProviderStatus.NO_CANDIDATE))
    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(_image(), providers=(provider,), fallback_adapter=fallback, request_id="r2")
    assert result.status is DetectionStatus.V52_FALLBACK
    assert result.corners == tuple(tuple(x) for x in original[0])
    assert len(seen) == 1


def test_legacy_fallback_can_be_disabled_for_explicit_v7_or_auto():
    provider = Provider(ProviderResult("synthetic", (), ProviderStatus.NO_CANDIDATE))
    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(_image(), providers=(provider,), allow_legacy_fallback=False, request_id="no-v52")
    assert result.status is DetectionStatus.ERROR
    assert "legacy_fallback_disabled" in result.error


def test_failed_provider_does_not_erase_successful_provider():
    good = Provider(ProviderResult("good", (candidate(),), ProviderStatus.SUCCESS))
    bad = Provider(ProviderResult("bad", (), ProviderStatus.TIMEOUT, timeout_code="deadline"))
    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(_image(), providers=(bad, good), request_id="r3")
    assert np.allclose(result.corners, candidate()["corners"], atol=2.0)
    assert result.status in {DetectionStatus.V7_RECOMMENDED, DetectionStatus.V7_LOW_CONFIDENCE}


def test_unsupported_input_falls_back_once():
    calls = []
    def fallback(image, **kwargs):
        calls.append(image)
        return ([[1, 2], [3, 4], [5, 6], [7, 8]], [0.2] * 4, [])
    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(object(), fallback_adapter=fallback, request_id="r4")
    assert result.status is DetectionStatus.V52_FALLBACK
    assert len(calls) == 1


def test_generic_no_primary_marker_on_nonblank_zero_candidates_falls_back():
    provider = Provider(ProviderResult("synthetic", (), ProviderStatus.NO_CANDIDATE,
                                       diagnostics={"primary_ambiguity": "no_primary"}))
    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(_image(), providers=(provider,),
                               fallback_adapter=lambda *args, **kwargs: ([[1, 2], [3, 4], [5, 6], [7, 8]], [0.2] * 4, []))
    assert result.status is DetectionStatus.V52_FALLBACK


def test_blank_zero_candidates_are_no_primary_without_fallback():
    provider = Provider(ProviderResult("synthetic", (), ProviderStatus.NO_CANDIDATE,
                                       diagnostics={"primary_ambiguity": "no_primary"}))
    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(np.zeros((100, 140, 3), dtype=np.uint8), providers=(provider,),
                               fallback_adapter=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    assert result.status is DetectionStatus.NO_PRIMARY_PHOTO


def test_border_completion_is_bounded_to_partial_background_frames():
    from photocut.algorithms.v7.detector import _border_completion_results

    partial = candidate("partial")
    partial = dict(partial)
    partial["corners"] = ((5, 25), (135, 25), (135, 70), (5, 70))
    partial["evidence"] = {"area_ratio": 0.30}
    background = ProviderResult("background", (partial,), ProviderStatus.SUCCESS)
    line = ProviderResult("lines", (partial,), ProviderStatus.SUCCESS)
    completed = _border_completion_results((background, line), 140, 100)
    assert len(completed[0].candidates) > 1
    assert completed[0].candidates[-1]["source"] == "background:frame_completion"
    assert completed[1] is line


def test_white_border_provider_cannot_change_primary_pool_but_remains_auditable():
    regular = candidate("regular", .62) | {"sources": ("lines:lsd",)}
    supplement = candidate("supplement", .99) | {
        "corners": ((5, 5), (134, 5), (134, 94), (5, 94)),
        "sources": ("background:border_connected",),
        "source": "background:border_connected",
        "evidence": {"supplemental_only": True},
    }
    base_provider = Provider(ProviderResult(
        "synthetic", (regular,), ProviderStatus.SUCCESS
    ))
    white_provider = Provider(ProviderResult(
        "white_border", (supplement,), ProviderStatus.SUCCESS
    ))

    from photocut.algorithms.v7.detector import detect_corners_v7
    result = detect_corners_v7(
        _image(), providers=(base_provider, white_provider),
        params=V7Parameters(scene_profile="scanner_white"),
        allow_legacy_fallback=False,
    )

    assert np.allclose(result.corners, regular["corners"], atol=2.0)
    supplement_audits = [
        audit for audit in result.candidate_audit
        if "background:border_connected" in audit.sources
    ]
    assert len(supplement_audits) == 1
    assert supplement_audits[0].truncation_stage == "selected"
    assert supplement_audits[0].stage_ranks["selected"] >= 6
    assert "scanner_boundary_score" in supplement_audits[0].stage_scores["components"]


def test_generic_profile_does_not_compute_scanner_boundary_evidence(monkeypatch):
    import photocut.algorithms.v7.detector as detector

    provider = Provider(ProviderResult(
        "synthetic", (candidate(),), ProviderStatus.SUCCESS
    ))
    monkeypatch.setattr(
        detector, "score_scanner_boundaries",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()),
    )

    result = detector.detect_corners_v7(
        _image(), providers=(provider,),
        params=V7Parameters(scene_profile="generic_single"),
    )

    assert result.status in {
        DetectionStatus.V7_RECOMMENDED, DetectionStatus.V7_LOW_CONFIDENCE,
    }
    assert all(
        "scanner_boundary_score" not in audit.stage_scores.get("components", {})
        for audit in result.candidate_audit
    )
