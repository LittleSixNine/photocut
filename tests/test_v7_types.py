import json

import numpy as np
import pytest

from photocut.algorithms.v7.types import (
    CandidateAudit,
    DetectionIdentity,
    DetectionResult,
    DetectionStatus,
    ProviderResult,
    ProviderStatus,
)


def identity(mode="safe"):
    return DetectionIdentity("req-1", "image-1", "rot0", "v7.0", "a" * 64, mode)


def quad(x=0.0):
    return ((x, x), (100.0 + x, x), (100.0 + x, 80.0 + x), (x, 80.0 + x))


def test_status_values_are_public_wire_values():
    assert [s.value for s in DetectionStatus] == [
        "v7_recommended", "v7_low_confidence", "v52_fallback",
        "no_primary_photo", "cancelled", "error",
    ]
    assert [s.value for s in ProviderStatus] == [
        "success", "no_candidate", "cancelled", "timeout", "budget_exhausted", "error",
    ]


def test_detection_result_round_trips_and_json_normalizes_numpy_scalars():
    audit = CandidateAudit(
        candidate_id="c1", sources=("contour",), original_legal_corners=quad(),
        pre_topk_corners=quad(), proposed_refined_corners=quad(.2),
        adopted_refined_corners=quad(.2), pre_truncation_risk_decisions=("none",),
        pre_truncation_risk_evidence={"edge": np.float32(.8)},
        stage_scores={"geometry": np.float64(.9)}, stage_ranks={"fused": 1},
        truncation_stage="none", truncation_reason=None,
    )
    result = DetectionResult(
        identity=identity(), status=DetectionStatus.V7_RECOMMENDED,
        corners=quad(), alternate_corners=None, overall_confidence=np.float32(.92),
        edge_confidences=(np.float64(.8),) * 4, corner_confidences=(.9,) * 4,
        risks=("none",), top1_sources=("contour",), alternate_sources=(),
        candidate_audit=(audit,), timings_ms={"total": np.float64(1.5)},
        debug={"scale": np.int64(800)}, error=None,
    )
    payload = result.to_dict()
    assert payload["status"] == "v7_recommended"
    json.dumps(payload)
    restored = DetectionResult.from_dict(payload)
    assert restored == result
    assert isinstance(restored.edge_confidences, tuple)


def test_result_and_audit_are_immutable_and_reject_truth_metrics():
    with pytest.raises((TypeError, ValueError)):
        CandidateAudit(candidate_id="x", sources=(), original_legal_corners=quad(),
                       pre_topk_corners=None, proposed_refined_corners=None,
                       adopted_refined_corners=None, pre_truncation_risk_decisions=(),
                       pre_truncation_risk_evidence={"TP": 1}, stage_scores={}, stage_ranks={})
    audit = CandidateAudit(candidate_id="x", sources=(), original_legal_corners=quad(),
                           pre_topk_corners=None, proposed_refined_corners=None,
                           adopted_refined_corners=None, pre_truncation_risk_decisions=(),
                           pre_truncation_risk_evidence={}, stage_scores={}, stage_ranks={})
    result = DetectionResult(identity(), DetectionStatus.ERROR, None, None, None, (), (), (), (), (), (audit,), {}, {}, "bad")
    with pytest.raises(Exception):
        result.risks = ("x",)


@pytest.mark.parametrize("metric_key", [
    "tp_count", "fn_count", "precision_score", "false_negative_rate",
    "nested_truth", "corner_iou", "accuracy_pct", "error_metric",
])
def test_audit_recursively_rejects_truth_metric_key_variants(metric_key):
    with pytest.raises((TypeError, ValueError)):
        CandidateAudit(
            candidate_id="x", sources=(), original_legal_corners=quad(),
            pre_truncation_risk_evidence={"nested": {metric_key: 1}},
            stage_scores={}, stage_ranks={},
        )


def test_provider_result_is_immutable_and_serializable():
    result = ProviderResult(
        provider="contour", candidates=({"id": "c1", "score": np.float32(.4)},),
        status=ProviderStatus.SUCCESS, elapsed_ms=np.float64(2.0),
        work_consumed=3, work_limit=10, timeout_code=None, error_code=None,
        diagnostics={"cache": True},
    )
    payload = result.to_dict()
    assert payload["status"] == "success"
    json.dumps(payload)
    assert ProviderResult.from_dict(payload) == result


def test_provider_budget_and_elapsed_constraints_and_stable_sets():
    with pytest.raises(ValueError):
        ProviderResult("x", status=ProviderStatus.SUCCESS, elapsed_ms=-1)
    with pytest.raises(ValueError):
        ProviderResult("x", status=ProviderStatus.SUCCESS, work_consumed=3, work_limit=2)
    assert ProviderResult("x", candidates=frozenset({"b", "a"})).candidates == ("a", "b")
    with pytest.raises(ValueError):
        ProviderResult("x", diagnostics={1: "a", "1": "b"})
