import json
from dataclasses import replace

import pytest

from photocut.algorithms.v7.types import ProviderStatus
from photocut.algorithms.v8.cascade import V8CascadeStatus, decide_v8_cascade
from photocut.algorithms.v8.code_seal import compute_validated_core_sha256
from photocut.algorithms.v8.evaluation import end_to_end_gate, summarize_end_to_end
from photocut.algorithms.v8.parameters import V8Parameters, canonical_sha256
from photocut.algorithms.v8.policy import V8Policy, load_v8_policy, write_v8_policy
from photocut.algorithms.v8.scanner_selector import (
    RankedScannerCandidate,
    ScannerExteriorEvidence,
    ScannerSelectionResult,
    ScannerSelectorConfig,
)


HASH = "sha256:" + "a" * 64


def _evidence(*, score=0.8, valid_sides=4):
    values = (score,) * 4
    return ScannerExteriorEvidence(
        score=score,
        side_scores=values,
        side_bed_scores=values,
        side_connected_scores=values,
        side_stability_scores=values,
        side_coverages=(1.0,) * 4,
        valid_side_count=valid_sides,
        side_score_min=score,
        side_score_mean=score,
        side_stability_mean=score,
        bed_lab=(250.0, 128.0, 128.0),
        bed_scale=10.0,
    )


def _selection(
    corners=((10, 10), (90, 10), (90, 90), (10, 90)),
    *,
    valid_sides=4,
    seed_support=1,
    seed_source_groups=1,
    exterior_score=0.8,
):
    selected = RankedScannerCandidate(
        candidate_id="edge-1",
        corners=tuple(corners),
        score=0.8,
        prior_score_normalized=0.7,
        area_ratio=0.64,
        evidence=_evidence(score=exterior_score, valid_sides=valid_sides),
        seed_support_count=seed_support,
        seed_source_group_count=seed_source_groups,
    )
    return ScannerSelectionResult(selected, "edge_base", "edge-1")


def _mask(corners):
    return {
        "candidate_id": "mask-1",
        "corners": corners,
        "evidence": {
            "foreground_probability": 0.95,
            "main_component_ratio": 0.97,
            "polygon_mask_iou": 0.91,
            "boundary_entropy": 0.50,
            "visible_margin": 0.01,
            "refinement": {"max_normalized_shift": 0.01},
        },
    }


def _policy(**changes):
    values = {
        "model_id": HASH,
        "model_sha256": HASH,
        "model_manifest_sha256": HASH,
        "runtime_config_sha256": HASH,
        "calibration_population_sha256": HASH,
        "calibration_run_sha256": HASH,
        "validated_core_sha256": HASH,
        "parameters": V8Parameters(),
    }
    values.update(changes)
    return V8Policy(**values)


def test_policy_identity_binds_every_parameter_and_rejects_tampering(tmp_path):
    policy = _policy()
    path = tmp_path / "policy.json"
    write_v8_policy(path, policy)

    loaded = load_v8_policy(path)
    assert loaded == policy
    assert loaded.policy_sha256 == policy.policy_sha256

    payload = json.loads(path.read_text())
    payload["parameters"]["selector_config"]["area_weight"] = 0.67
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="policy SHA-256"):
        load_v8_policy(path)


def test_legacy_v8_policy_payload_remains_loadable_for_audit():
    parameters = V8Parameters().to_dict()
    parameters["schema_version"] = 1
    parameters["evidence_schema"] = "v8-scanner-evidence-v1"
    for name in (
        "edge_support_exterior_tolerance",
        "minimum_unopposed_exterior_score",
        "edge_cluster_distance",
        "minimum_automatic_score_margin",
        "minimum_edge_seed_support",
        "minimum_edge_source_group_support",
    ):
        parameters["selector_config"].pop(name)
    parameters["selector_config"]["minimum_automatic_valid_sides"] = 1
    payload = {
        "schema_version": 1,
        "policy_id": "v8-auto-v1",
        "algorithm_version": "8.0",
        "production_promotion": False,
        "model_id": HASH,
        "model_sha256": HASH,
        "model_manifest_sha256": HASH,
        "runtime_config_sha256": HASH,
        "calibration_population_sha256": HASH,
        "calibration_run_sha256": HASH,
        "validated_core_sha256": HASH,
        "parameters": parameters,
    }
    value = {**payload, "policy_sha256": canonical_sha256(payload)}

    try:
        loaded = V8Policy.from_dict(value)
    except ValueError as exc:
        pytest.fail(f"legacy V8 policy did not load: {exc}")

    assert loaded.to_dict() == value


def test_policy_is_research_only_scanner_white_v8():
    assert _policy().schema_version == 2
    assert _policy().policy_id == "v8-auto-v2"
    assert _policy().algorithm_version == "8.1"
    with pytest.raises(ValueError, match="production promotion"):
        replace(_policy(), production_promotion=True)
    with pytest.raises(ValueError, match="scanner_white"):
        replace(_policy(), parameters=replace(V8Parameters(), scene_profile="generic_single"))


def test_v82_policy_binds_boundary_quality_artifact_and_preserves_v81_schema():
    policy = _policy(
        schema_version=3,
        policy_id="v8-auto-v3",
        algorithm_version="8.2",
        boundary_quality_artifact_sha256=HASH,
    )

    payload = policy.to_dict()
    loaded = V8Policy.from_dict(payload)

    assert loaded == policy
    assert loaded.boundary_quality_artifact_sha256 == HASH
    assert "boundary_quality_artifact_sha256" not in _policy().to_dict()
    with pytest.raises(ValueError, match="boundary-quality artifact"):
        _policy(
            schema_version=3,
            policy_id="v8-auto-v3",
            algorithm_version="8.2",
        )


def test_parameter_identity_changes_when_selector_threshold_changes():
    original = V8Parameters()
    changed = replace(
        original,
        selector_config=replace(
            original.selector_config,
            v7_edge_agreement_distance=0.019,
        ),
    )
    assert original.sha256() != changed.sha256()
    assert original.to_dict()["selector_config"] != changed.to_dict()["selector_config"]


@pytest.mark.parametrize(
    "change",
    (
        {"exterior_weight": -0.1},
        {"area_weight": 1.2},
        {"v7_edge_agreement_ratio": 0.5},
        {"manual_conflict_distance": 0.5},
    ),
)
def test_selector_config_rejects_unbounded_policy_values(change):
    with pytest.raises((TypeError, ValueError)):
        ScannerSelectorConfig(**change)


def test_state_table_accepts_consensus_and_records_policy_identity():
    corners = ((10, 10), (90, 10), (90, 90), (10, 90))
    policy = _policy()
    decision = decide_v8_cascade(
        _selection(corners),
        provider_status=ProviderStatus.SUCCESS,
        policy=policy,
        image_size=(100, 100),
        v7_candidate={"candidate_id": "v7", "corners": corners},
        mask_candidate=_mask(corners),
    )
    assert decision.status is V8CascadeStatus.AUTOMATIC
    assert decision.policy_sha256 == policy.policy_sha256
    assert decision.reason == "edge_base"
    assert decision.evidence["v7_mask_distance"] == pytest.approx(0.0)


def test_state_table_uses_v7_fallback_when_v7_is_the_only_candidate():
    corners = ((10, 10), (90, 10), (90, 90), (10, 90))
    decision = decide_v8_cascade(
        replace(_selection(corners), reason="v7_only"),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={"candidate_id": "edge-1", "corners": corners},
    )

    assert decision.status is V8CascadeStatus.V7_FALLBACK
    assert decision.reason == "v7_only_candidate"


def test_state_table_rejects_v7_mask_rescue_when_mask_is_weak():
    corners = ((10, 10), (90, 10), (90, 90), (10, 90))
    weak_mask = _mask(corners)
    weak_mask["evidence"]["foreground_probability"] = 0.10
    decision = decide_v8_cascade(
        replace(_selection(corners), reason="v7_mask_consensus_rescue"),
        provider_status=ProviderStatus.SUCCESS,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={"candidate_id": "edge-1", "corners": corners},
        mask_candidate=weak_mask,
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "weak_mask_evidence"


def test_state_table_rejects_mask_only_selection_without_independent_support():
    corners = ((10, 10), (90, 10), (90, 90), (10, 90))
    mask = _mask(corners)
    selection = replace(
        _selection(corners),
        selected=replace(_selection(corners).selected, candidate_id="mask-1"),
        reason="mask_only",
    )
    decision = decide_v8_cascade(
        selection,
        provider_status=ProviderStatus.SUCCESS,
        policy=_policy(),
        image_size=(100, 100),
        mask_candidate=mask,
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "mask_only_unconfirmed"


def test_state_table_sends_three_way_strong_conflict_to_manual_review():
    policy = _policy()
    decision = decide_v8_cascade(
        _selection(),
        provider_status=ProviderStatus.SUCCESS,
        policy=policy,
        image_size=(100, 100),
        v7_candidate={
            "candidate_id": "v7",
            "corners": ((0, 0), (45, 0), (45, 45), (0, 45)),
        },
        mask_candidate=_mask(((55, 55), (99, 55), (99, 99), (55, 99))),
    )
    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "strong_three_way_conflict"


def test_state_table_rejects_v7_edge_conflict_without_strong_mask_support():
    decision = decide_v8_cascade(
        _selection(seed_support=3, seed_source_groups=2),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={
            "candidate_id": "v7",
            "corners": ((0, 0), (45, 0), (45, 45), (0, 45)),
        },
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "v7_edge_unresolved_conflict"


def test_state_table_rejects_low_rank_v7_edge_rescue_when_strong_mask_disagrees():
    corners = ((10, 10), (90, 10), (90, 90), (10, 90))
    selection = replace(
        _selection(corners, seed_support=2, seed_source_groups=2),
        reason="v7_edge_agreement_rescue",
        competing_edge_candidate_id="edge-higher-ranked",
        competing_edge_distance=0.09,
        edge_score_margin=-0.04,
        selected_edge_cluster_size=3,
        selected_edge_cluster_seed_support_count=2,
        selected_edge_cluster_source_group_count=2,
    )
    decision = decide_v8_cascade(
        selection,
        provider_status=ProviderStatus.SUCCESS,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={"candidate_id": "v7", "corners": corners},
        mask_candidate=_mask(((0, 0), (45, 0), (45, 45), (0, 45))),
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "unstable_v7_edge_rescue"


def test_state_table_keeps_low_rank_v7_edge_rescue_when_mask_conflict_is_mild():
    corners = ((10, 10), (90, 10), (90, 90), (10, 90))
    mildly_shifted_mask = tuple((x + 5, y) for x, y in corners)
    selection = replace(
        _selection(corners, seed_support=2, seed_source_groups=2),
        reason="v7_edge_agreement_rescue",
        competing_edge_candidate_id="edge-higher-ranked",
        competing_edge_distance=0.09,
        edge_score_margin=-0.04,
        selected_edge_cluster_size=3,
        selected_edge_cluster_seed_support_count=2,
        selected_edge_cluster_source_group_count=2,
    )
    decision = decide_v8_cascade(
        selection,
        provider_status=ProviderStatus.SUCCESS,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={"candidate_id": "v7", "corners": corners},
        mask_candidate=_mask(mildly_shifted_mask),
    )

    assert decision.status is V8CascadeStatus.AUTOMATIC


def test_state_table_keeps_border_touching_candidate_with_two_visible_sides():
    corners = ((0, 0), (99, 0), (90, 90), (5, 90))
    decision = decide_v8_cascade(
        _selection(corners, valid_sides=2),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={"candidate_id": "v7", "corners": corners},
    )
    assert decision.status is V8CascadeStatus.AUTOMATIC


def test_state_table_accepts_one_visible_side_when_v7_independently_agrees():
    corners = ((0, 0), (99, 0), (99, 99), (0, 99))
    decision = decide_v8_cascade(
        _selection(corners, valid_sides=1),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
        v7_candidate={"candidate_id": "v7", "corners": corners},
    )

    assert decision.status is V8CascadeStatus.AUTOMATIC


def test_state_table_rejects_one_visible_side_without_independent_agreement():
    corners = ((0, 0), (99, 0), (99, 99), (0, 99))
    decision = decide_v8_cascade(
        _selection(corners, valid_sides=1),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "insufficient_scanner_exterior_evidence"


def test_state_table_rejects_single_seed_edge_without_independent_support():
    decision = decide_v8_cascade(
        _selection(
            seed_support=1,
            seed_source_groups=1,
            exterior_score=0.30,
        ),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "unsupported_single_source_selection"


def test_state_table_accepts_single_source_edge_with_strong_exterior_evidence():
    decision = decide_v8_cascade(
        _selection(
            seed_support=1,
            seed_source_groups=1,
            exterior_score=0.70,
        ),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.AUTOMATIC


def test_state_table_accepts_multi_source_support_aggregated_by_edge_cluster():
    selection = replace(
        _selection(seed_support=1, seed_source_groups=1),
        selected_edge_cluster_size=2,
        selected_edge_cluster_seed_support_count=2,
        selected_edge_cluster_source_group_count=2,
    )

    decision = decide_v8_cascade(
        selection,
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.AUTOMATIC


def test_state_table_rejects_weak_edge_even_with_multiple_seed_support():
    decision = decide_v8_cascade(
        _selection(
            seed_support=3,
            seed_source_groups=2,
            exterior_score=0.30,
        ),
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "weak_unopposed_edge_evidence"


def test_state_table_accepts_weak_exterior_with_clear_distant_competitor_margin():
    selection = replace(
        _selection(
            seed_support=3,
            seed_source_groups=2,
            exterior_score=0.30,
        ),
        competing_edge_candidate_id="edge-2",
        competing_edge_distance=0.08,
        edge_score_margin=0.09,
    )

    decision = decide_v8_cascade(
        selection,
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.AUTOMATIC


def test_state_table_rejects_low_margin_distant_edge_competitor():
    selection = replace(
        _selection(seed_support=3, seed_source_groups=2),
        competing_edge_candidate_id="edge-2",
        competing_edge_distance=0.08,
        edge_score_margin=0.005,
    )

    decision = decide_v8_cascade(
        selection,
        provider_status=ProviderStatus.NO_CANDIDATE,
        policy=_policy(),
        image_size=(100, 100),
    )

    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "unstable_edge_ranking"


def test_state_table_rejects_insufficient_exterior_evidence():
    decision = decide_v8_cascade(
        _selection(valid_sides=0),
        provider_status=ProviderStatus.SUCCESS,
        policy=_policy(),
        image_size=(100, 100),
    )
    assert decision.status is V8CascadeStatus.MANUAL_REVIEW
    assert decision.reason == "insufficient_scanner_exterior_evidence"


@pytest.mark.parametrize(
    "status",
    (ProviderStatus.ERROR, ProviderStatus.TIMEOUT, ProviderStatus.BUDGET_EXHAUSTED),
)
def test_state_table_falls_back_to_v7_when_model_execution_fails(status):
    decision = decide_v8_cascade(
        _selection(),
        provider_status=status,
        policy=_policy(),
        image_size=(100, 100),
    )
    assert decision.status is V8CascadeStatus.V7_FALLBACK
    assert decision.reason == f"mask_provider_{status.value}"


def test_state_table_preserves_cancellation_as_a_distinct_terminal_state():
    decision = decide_v8_cascade(
        _selection(),
        provider_status=ProviderStatus.CANCELLED,
        policy=_policy(),
        image_size=(100, 100),
    )
    assert decision.status is V8CascadeStatus.CANCELLED
    assert decision.reason == "mask_provider_cancelled"


def test_validated_core_hash_changes_on_any_core_byte_and_rejects_symlink(tmp_path):
    root = tmp_path / "project"
    (root / "algorithms" / "v5_2").mkdir(parents=True)
    (root / "algorithms" / "v7").mkdir()
    (root / "algorithms" / "v8").mkdir()
    for relative in (
        "core.py",
        "algorithms/v5_2/detector.py",
        "config.py",
    ):
        (root / relative).write_text(relative)
    (root / "algorithms" / "v7" / "a.py").write_text("one")
    (root / "algorithms" / "v8" / "b.py").write_text("two")

    first = compute_validated_core_sha256(root)
    (root / "algorithms" / "v8" / "b.py").write_text("three")
    second = compute_validated_core_sha256(root)
    assert first != second

    (root / "algorithms" / "v8" / "linked.py").symlink_to(
        root / "algorithms" / "v7" / "a.py"
    )
    with pytest.raises(ValueError, match="symlink"):
        compute_validated_core_sha256(root)


def test_end_to_end_summary_uses_entire_population_and_real_terminal_states():
    rows = [
        {"decision_status": "automatic", "automatic_strict": True, "automatic_catastrophic": False},
        {"decision_status": "v7_fallback", "automatic_strict": True, "automatic_catastrophic": False},
        {"decision_status": "automatic", "automatic_strict": False, "automatic_catastrophic": False},
        {"decision_status": "manual_review", "automatic_strict": False, "automatic_catastrophic": False},
    ]
    summary = summarize_end_to_end(rows)
    assert summary == {
        "sample_count": 4,
        "automatic_count": 3,
        "automatic_strict_count": 2,
        "automatic_strict_rate": 0.5,
        "automatic_catastrophic_count": 0,
        "manual_or_rescan_count": 1,
        "manual_or_rescan_rate": 0.25,
        "v7_fallback_count": 1,
        "decision_status_counts": {
            "automatic": 2,
            "manual_review": 1,
            "v7_fallback": 1,
        },
    }
    gate = end_to_end_gate(
        summary, frozen_accessed=False, source_modified_count=0,
    )
    assert gate["passed"] is False
    assert "automatic_strict_at_least_95_percent" in gate["failed_conditions"]


def test_end_to_end_summary_does_not_label_truth_failure_as_manual():
    summary = summarize_end_to_end(({
        "decision_status": "automatic",
        "automatic_strict": False,
        "automatic_catastrophic": False,
    },))
    assert summary["automatic_count"] == 1
    assert summary["manual_or_rescan_count"] == 0
