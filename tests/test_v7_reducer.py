import math

import pytest

from photocut.algorithms.v7.reducer import FusedCandidates, TopKSelection, fuse_candidates, select_topk
from photocut.algorithms.v7.types import ProviderResult, ProviderStatus


def quad(x=10.0, y=10.0, width=70.0, height=50.0):
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))


def candidate(cid, corners, *, source="detector", score=0.5, **extra):
    return {"id": cid, "candidate_id": cid, "corners": corners, "source": source,
            "sources": (source,), "score": score, **extra}


def providers(*pairs):
    return [{"provider": provider, "candidates": values} for provider, values in pairs]


def test_fuse_rejects_malformed_geometry_and_reports_reasons():
    bad = [
        candidate("nan", ((0, 0), (1, 0), (math.nan, 1), (0, 1))),
        candidate("duplicate", ((0, 0), (10, 0), (10, 0), (0, 10))),
        candidate("concave", ((0, 0), (20, 0), (5, 5), (0, 20))),
        candidate("small", quad(width=1, height=1)),
    ]
    fused = fuse_candidates(providers(("p1", bad)), image_size=(100, 100))
    assert isinstance(fused, FusedCandidates)
    assert fused.candidates == ()
    assert set(fused.rejected_ids) == {"nan", "duplicate", "concave", "small"}
    assert all(fused.rejected_reasons[cid] for cid in fused.rejected_ids)


def test_fuse_audits_non_mapping_malformed_provider_items_instead_of_crashing():
    fused = fuse_candidates(providers(("broken", [None, "not-a-candidate"])), image_size=(100, 100))
    assert fused.candidates == ()
    assert len(fused.rejected_ids) == 2
    assert all(reason == "malformed_candidate" for reason in fused.rejected_reasons.values())


def test_fuse_discards_partial_candidates_from_failed_provider_status():
    good = ProviderResult("good", (candidate("good-id", quad()),), status=ProviderStatus.SUCCESS)
    failed = ProviderResult("failed", (candidate("partial-id", quad(x=150)),), status=ProviderStatus.TIMEOUT)
    fused = fuse_candidates([good, failed], image_size=(300, 200))
    assert {item["candidate_id"] for item in fused.candidates} == {"good-id"}
    assert fused.rejected_reasons["partial-id"] == "provider_status:timeout"


def test_fuse_rejects_inconsistent_no_candidate_status_with_partial_output():
    inconsistent = ProviderResult("bad", (candidate("partial-no", quad()),), status=ProviderStatus.NO_CANDIDATE)
    fused = fuse_candidates([inconsistent], image_size=(100, 100))
    assert fused.candidates == ()
    assert fused.rejected_reasons["partial-no"] == "provider_status:no_candidate"


def test_fuse_enforces_hard_limit_of_forty_candidates():
    with pytest.raises(ValueError, match="40"):
        fuse_candidates([], fused_limit=41)


def test_fuse_reports_all_failed_provider_statuses_when_no_successful_provider_exists():
    failed = ProviderResult("failed", (candidate("partial-only", quad()),), status=ProviderStatus.ERROR)
    fused = fuse_candidates([failed], image_size=(100, 100))
    assert fused.candidates == ()
    assert fused.rejected_reasons["partial-only"] == "provider_status:error"


def test_fuse_rejects_candidates_without_image_size_for_bounds_validation():
    fused = fuse_candidates(providers(("p", [candidate("no-size", quad())])))
    assert fused.candidates == ()
    assert fused.rejected_reasons["no-size"] == "missing_image_size"


def test_select_topk_traces_duplicate_candidate_ids_in_audited_input():
    first = candidate("same", quad(), source="p", pre_score=.6, audit={"outer_risk": "none"})
    second = candidate("same", quad(x=150), source="q", pre_score=.5, audit={"outer_risk": "none"})
    result = select_topk([first, second], image_size=(300, 200), top_k=2)
    assert [item["candidate_id"] for item in result.candidates] == ["same"]
    assert {event["reason"] for event in result.truncation_trace} == {"duplicate_candidate_id"}


def test_select_topk_rejects_invalid_candidate_id_and_uses_nested_pre_score_total():
    invalid = candidate("", quad(), pre_score=.9, audit={"outer_risk": "none"})
    high = candidate("high", quad(x=150), audit={"outer_risk": "none"},
                     stage_scores={"pre_score_total": .8})
    result = select_topk([invalid, high], image_size=(300, 200), top_k=1)
    assert [item["candidate_id"] for item in result.candidates] == ["high"]
    assert any(event["reason"] == "invalid_candidate_id" for event in result.truncation_trace)


def test_fuse_merges_cross_provider_duplicates_and_retains_all_provenance():
    q = quad()
    fused = fuse_candidates(
        providers(("background", [candidate("b", q, source="mask")]),
                  ("lines", [candidate("l", tuple((x + 0.2, y - 0.1) for x, y in q), source="lsd")])) ,
        image_size=(100, 100), dedup_distance=0.01,
    )
    assert len(fused.candidates) == 1
    merged = fused.candidates[0]
    assert set(merged["providers"]) == {"background", "lines"}
    assert set(merged["sources"]) == {"mask", "lsd"}
    assert {item["candidate_id"] for item in merged["provenance"]} == {"b", "l"}


def test_fuse_preserves_rejection_reasons_for_repeated_ids_across_providers():
    q = quad()
    fused = fuse_candidates(providers(("a", [candidate("same", q)]),
                                      ("b", [candidate("same", quad(x=150))]),
                                      ("c", [candidate("same", quad(x=220))])),
                            image_size=(300, 200))
    assert len(fused.candidates) == 1
    assert isinstance(fused.rejected_reasons["same"], tuple)
    assert fused.rejected_reasons["same"].count("duplicate_id") == 2


def test_fair_round_robin_caps_flooding_and_is_stable_under_input_reordering():
    flood = [candidate(f"a{i:02d}", quad(x=5 + i * 75), source="a", score=1 - i / 100) for i in range(80)]
    other = [candidate(f"b{i:02d}", quad(x=5 + i * 75, y=80), source="b") for i in range(40)]
    loose = type("P", (), {"min_edge_ratio": .001, "min_area_ratio": .001})()
    first = fuse_candidates(providers(("flood", flood), ("other", other)), image_size=(7000, 200), fused_limit=40, dedup_distance=.001, params=loose)
    second = fuse_candidates(providers(("other", list(reversed(other))), ("flood", list(reversed(flood)))), image_size=(7000, 200), fused_limit=40, dedup_distance=.001, params=loose)
    assert len(first.candidates) == 40
    assert len([c for c in first.candidates if "flood" in c["providers"]]) <= 20
    assert len([c for c in first.candidates if "other" in c["providers"]]) >= 8
    assert [c["candidate_id"] for c in first.candidates] == [c["candidate_id"] for c in second.candidates]


def test_select_topk_requires_audit_and_keeps_source_and_geometry_diversity():
    items = []
    for i in range(10):
        items.append(candidate(f"d{i}", quad(x=10 + i * 0.03, y=10), source="flood", score=0.99 - i * .001,
                               pre_score=0.99 - i * .001, audit={"outer_risk": "none"}))
    items.append(candidate("diverse", quad(x=150, y=20), source="lines", score=.5,
                           pre_score=.5, audit={"outer_risk": "none"}))
    items.append(candidate("third", quad(x=20, y=100), source="contour", score=.4,
                           pre_score=.4, audit={"outer_risk": "none"}))
    result = select_topk(items, image_size=(300, 200), top_k=5)
    assert isinstance(result, TopKSelection)
    assert len(result.candidates) == 5
    assert "diverse" in {c["candidate_id"] for c in result.candidates}
    assert "third" in {c["candidate_id"] for c in result.candidates}
    assert result.truncation_trace
    assert all("candidate_id" in event and "reason" in event for event in result.truncation_trace)


def test_select_topk_has_stable_tie_break_and_rejects_unaudited_input():
    a = candidate("a", quad(x=10), source="p", pre_score=.5, audit={"outer_risk": "none"})
    b = candidate("b", quad(x=150), source="q", pre_score=.5, audit={"outer_risk": "none"})
    one = select_topk([b, a], image_size=(300, 200), top_k=2)
    two = select_topk([a, b], image_size=(300, 200), top_k=2)
    assert [c["candidate_id"] for c in one.candidates] == [c["candidate_id"] for c in two.candidates] == ["a", "b"]
    with pytest.raises(ValueError, match="audit"):
        select_topk([candidate("raw", quad())], image_size=(100, 100), top_k=1)


def test_select_topk_one_slot_always_uses_global_score_before_source_diversity():
    low = candidate("a-low", quad(), source="a", pre_score=.1, audit={"outer_risk": "none"})
    high = candidate("z-high", quad(x=150), source="z", pre_score=.9, audit={"outer_risk": "none"})
    result = select_topk([low, high], image_size=(300, 200), top_k=1)
    assert [item["candidate_id"] for item in result.candidates] == ["z-high"]


def test_select_topk_records_audit_precondition_rejections():
    valid = candidate("valid", quad(), source="p", pre_score=.5, audit={"outer_risk": "none"})
    missing_score = candidate("missing-score", quad(x=150), source="q", score=None,
                              audit={"outer_risk": "none"})
    empty_audit = candidate("empty-audit", quad(x=220), source="r", pre_score=.9, audit={})
    result = select_topk([valid, missing_score, empty_audit], image_size=(400, 200), top_k=2)
    reasons = {event["candidate_id"]: event["reason"] for event in result.truncation_trace}
    assert reasons["missing-score"] == "missing_or_nonfinite_pre_score"
    assert reasons["empty-audit"] == "missing_risk_audit"


def test_select_topk_uses_later_finite_score_when_first_score_field_is_invalid():
    item = candidate("fallback-score", quad(), source="p", pre_score=None,
                     cheap_score=.4, audit={"outer_risk": "none"})
    result = select_topk([item], image_size=(100, 100), top_k=1)
    assert [entry["candidate_id"] for entry in result.candidates] == ["fallback-score"]


def test_select_topk_rejects_nonfinite_or_negative_dedup_distance():
    item = candidate("valid", quad(), pre_score=.5, audit={"outer_risk": "none"})
    with pytest.raises(ValueError, match="dedup_distance"):
        select_topk([item], image_size=(100, 100), dedup_distance=float("nan"))
    with pytest.raises(ValueError, match="dedup_distance"):
        select_topk([item], image_size=(100, 100), dedup_distance=-1)
