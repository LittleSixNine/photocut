import numpy as np

from photocut.algorithms.v7.features import ImageFeatureContext
from photocut.algorithms.v7.scoring import score_candidates, score_components
from photocut.algorithms.v7.types import CandidateAudit


def quad(x=20, y=20, w=160, h=110):
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


def candidate(cid, corners, source="p", score=0.5):
    return {"candidate_id": cid, "corners": corners, "sources": (source,), "score": score}


def image_with_photo():
    image = np.full((180, 220, 3), 232, np.uint8)
    image[25:140, 25:185] = (30, 80, 160)
    image[25:28, 25:185] = 10
    image[137:140, 25:185] = 10
    image[25:140, 25:28] = 10
    image[25:140, 182:185] = 10
    return image


def test_component_scores_are_bounded_and_weak_edge_is_not_hidden():
    image = image_with_photo()
    context = ImageFeatureContext(image)
    good = score_components(candidate("good", quad(25, 25, 160, 115)), context, image_size=(220, 180))
    weak = score_components(candidate("weak", ((25, 25), (185, 25), (185, 40), (25, 140))), context, image_size=(220, 180))
    expected = {"geometry", "gradient_continuity", "gradient_direction", "inside_outside_lab",
                "inside_outside_texture", "outside_background_consistency", "internal_content_coverage",
                "provider_consensus", "candidate_margin", "nesting"}
    assert expected.issubset(good)
    assert all(0 <= float(good[name]) <= 1 for name in expected)
    assert float(weak["edge_score"]) <= max(float(good["edge_score"]), 0.75)
    assert float(weak["pre_score"]) < float(good["pre_score"])


def test_scanner_boundary_score_prefers_true_outer_edge_to_printed_inner_edge():
    # Speckled scanner bed, smooth white paper border, and coloured print area.
    yy, xx = np.indices((220, 280))
    bed = 238 + ((xx * 7 + yy * 11) % 13)
    image = np.repeat(bed[..., None], 3, axis=2).astype(np.uint8)
    image[25:195, 30:250] = 224
    image[45:175, 50:230] = (30, 90, 170)

    outer = score_components(
        candidate("outer", quad(30, 25, 220, 170)), image,
        image_size=(280, 220),
    )
    inner = score_components(
        candidate("inner", quad(50, 45, 180, 130)), image,
        image_size=(280, 220),
    )

    assert 0 <= outer["scanner_boundary_score"] <= 1
    assert outer["scanner_boundary_score"] > inner["scanner_boundary_score"]
    assert len(outer["scanner_boundary_per_edge"]) == 4


def test_independent_provider_agreement_breaks_near_ties():
    image = np.full((180, 220, 3), 128, dtype=np.uint8)
    agreement = score_components({
        "candidate_id": "agreement",
        "corners": quad(8, 8, 204, 164),
        "sources": ("background:min_area_rect", "lines:lsd"),
    }, image, image_size=(220, 180))
    single = score_components({
        "candidate_id": "single",
        "corners": quad(8, 8, 204, 164),
        "sources": ("lines:lsd",),
    }, image, image_size=(220, 180))
    assert agreement["provider_consensus"] > single["provider_consensus"]
    assert agreement["pre_score"] > single["pre_score"]


def test_frame_extent_prevents_small_internal_rectangle_from_winning():
    image = image_with_photo()
    result = score_candidates([
        candidate("outer", quad(25, 25, 160, 115), source="contour"),
        candidate("internal", quad(60, 50, 80, 60), source="lines"),
    ], image, image_size=(220, 180), top_k=1)
    assert result.selected[0]["candidate_id"] == "outer"
    assert result.selected[0]["component_scores"]["frame_extent"] > 0


def test_every_fused_candidate_gets_candidate_audit_before_topk_truncation():
    image = image_with_photo()
    items = [candidate(f"c{i}", quad(25 + i * 2, 25 + i, 160, 115), source="p", score=1 - i / 50) for i in range(8)]
    result = score_candidates(items, image, image_size=(220, 180), top_k=5)
    assert len(result.audits) == len(items)
    assert all(isinstance(audit, CandidateAudit) for audit in result.audits)
    assert len(result.selected) <= 5
    assert all("pre_score" in audit.stage_scores for audit in result.audits)
    serialized = [audit.to_dict() for audit in result.audits]
    assert not any(key.lower() in {"tp", "fn", "truth", "groundtruth"} for item in serialized for key in str(item).split())


def test_feature_cache_is_reused_for_forty_candidates():
    context = ImageFeatureContext(image_with_photo())
    items = [candidate(str(i), quad(25 + (i % 8), 25 + (i // 8), 150, 105)) for i in range(40)]
    score_candidates(items, context, image_size=(220, 180), top_k=5, scale=160)
    assert context.build_counts["gradient"] == 1
    assert context.build_counts["lab"] == 1
    assert context.build_counts["bgr"] == 1


def test_full_edge_pass_is_limited_to_top_five(monkeypatch):
    import photocut.algorithms.v7.scoring as scoring
    context = ImageFeatureContext(image_with_photo())
    items = [candidate(str(i), quad(25 + i, 25, 150, 105)) for i in range(40)]
    calls = {"count": 0}
    original = scoring._edge_pass
    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(scoring, "_edge_pass", counted)
    result = score_candidates(items, context, image_size=(220, 180), top_k=10)
    assert len(result.selected) <= 5
    assert calls["count"] <= 5


def test_global_gradient_statistics_are_built_once_per_scale(monkeypatch):
    import photocut.algorithms.v7.scoring as scoring
    context = ImageFeatureContext(image_with_photo())
    items = [candidate(str(i), quad(25 + (i % 8), 25 + (i // 8), 150, 105)) for i in range(40)]
    calls = {"count": 0}
    original = scoring.np.percentile
    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(scoring.np, "percentile", counted)
    score_candidates(items, context, image_size=(220, 180), top_k=5)
    assert calls["count"] <= 2


def test_nested_candidate_updates_outer_pre_score_and_risk_audit():
    image = image_with_photo()
    result = score_candidates([
        candidate("outer", quad(0, 0, 219, 179)),
        candidate("inner", quad(25, 25, 160, 115)),
    ], image, image_size=(220, 180), top_k=2)
    outer = next(a for a in result.audits if a.candidate_id == "outer")
    assert outer.pre_truncation_risk_evidence["nested_candidate_exists"] is True
    # The main scorer keeps the archived raw nested-child score.  The current
    # V7.1 change is scanner-only and must not revive the abandoned ranker.
    assert np.isclose(
        outer.stage_scores["components"]["nesting"],
        outer.pre_truncation_risk_evidence["nested_candidate_score"],
    )


def test_rotated_parent_requires_polygon_containment_not_bbox_only():
    image = image_with_photo()
    diamond = ((110, 0), (219, 90), (110, 179), (0, 90))
    axis_child = ((40, 55), (180, 55), (180, 125), (40, 125))
    result = score_candidates([
        {"candidate_id": "diamond", "corners": diamond, "sources": ("shape",)},
        {"candidate_id": "axis", "corners": axis_child, "sources": ("lines",)},
    ], image, image_size=(220, 180), top_k=2)
    outer = next(a for a in result.audits if a.candidate_id == "diamond")
    assert outer.pre_truncation_risk_evidence["nested_candidate_exists"] is False
