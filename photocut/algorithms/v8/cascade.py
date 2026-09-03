"""Truth-independent V8 state table and adapter result contracts."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from photocut.algorithms.v7.types import ProviderStatus

from .policy import V8Policy
from .scanner_selector import ScannerSelectionResult


class V8CascadeStatus(str, Enum):
    AUTOMATIC = "automatic"
    MANUAL_REVIEW = "manual_review"
    V7_FALLBACK = "v7_fallback"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class V8CascadeDecision:
    status: V8CascadeStatus
    reason: str
    policy_sha256: str
    provider_status: ProviderStatus
    selected_candidate_id: str | None
    selected_corners: tuple[tuple[float, float], ...] | None
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "policy_sha256": self.policy_sha256,
            "provider_status": self.provider_status.value,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_corners": None if self.selected_corners is None else [
                list(point) for point in self.selected_corners
            ],
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class V8DormantResult:
    core_payload: Mapping[str, Any]
    audit_envelope: Mapping[str, Any]
    status: V8CascadeStatus


def _corners(candidate: Mapping[str, Any] | None) -> tuple[tuple[float, float], ...] | None:
    if candidate is None:
        return None
    value = candidate.get("corners")
    if not isinstance(value, (tuple, list)) or len(value) != 4:
        return None
    try:
        result = tuple((float(point[0]), float(point[1])) for point in value)
    except (TypeError, ValueError, IndexError, OverflowError):
        return None
    if any(not math.isfinite(x) or not math.isfinite(y) for x, y in result):
        return None
    return result


def _distance(
    left: Sequence[Sequence[float]] | None,
    right: Sequence[Sequence[float]] | None,
    image_size: tuple[int, int],
) -> float | None:
    if left is None or right is None:
        return None
    return float(
        np.max(np.linalg.norm(np.asarray(left) - np.asarray(right), axis=1))
        / math.hypot(*image_size)
    )


def _strong_mask(candidate: Mapping[str, Any] | None, policy: V8Policy) -> bool:
    if candidate is None:
        return False
    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    try:
        refinement = evidence.get("refinement", {})
        shift = (
            float(refinement.get("max_normalized_shift", 0.0))
            if isinstance(refinement, Mapping)
            else math.inf
        )
        params = policy.parameters
        return (
            float(evidence.get("foreground_probability", 0.0))
            >= float(params.mask_min_foreground_probability)
            and float(evidence.get("main_component_ratio", 0.0))
            >= float(params.mask_min_component_ratio)
            and float(evidence.get("polygon_mask_iou", 0.0))
            >= float(params.mask_min_polygon_iou)
            and float(evidence.get("boundary_entropy", 1.0))
            <= float(params.mask_max_boundary_entropy)
            and shift <= float(params.mask_max_refinement_shift)
        )
    except (TypeError, ValueError, OverflowError):
        return False


def decide_v8_cascade(
    selection: ScannerSelectionResult | None,
    *,
    provider_status: ProviderStatus,
    policy: V8Policy,
    image_size: tuple[int, int],
    v7_candidate: Mapping[str, Any] | None = None,
    mask_candidate: Mapping[str, Any] | None = None,
) -> V8CascadeDecision:
    """Apply the sealed state table without running detection or using truth."""
    if not isinstance(policy, V8Policy):
        raise TypeError("policy must be V8Policy")
    if not isinstance(provider_status, ProviderStatus):
        raise TypeError("provider_status must be ProviderStatus")
    if (
        not isinstance(image_size, tuple)
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ValueError("image_size must contain positive integers")
    if provider_status is ProviderStatus.CANCELLED:
        return V8CascadeDecision(
            V8CascadeStatus.CANCELLED,
            "mask_provider_cancelled",
            policy.policy_sha256,
            provider_status,
            None,
            None,
            {},
        )
    if provider_status in {
        ProviderStatus.ERROR,
        ProviderStatus.TIMEOUT,
        ProviderStatus.BUDGET_EXHAUSTED,
    }:
        return V8CascadeDecision(
            V8CascadeStatus.V7_FALLBACK,
            f"mask_provider_{provider_status.value}",
            policy.policy_sha256,
            provider_status,
            None,
            None,
            {},
        )
    if selection is None:
        return V8CascadeDecision(
            V8CascadeStatus.V7_FALLBACK,
            "no_v8_candidate",
            policy.policy_sha256,
            provider_status,
            None,
            None,
            {},
        )
    if selection.reason == "v7_only":
        return V8CascadeDecision(
            V8CascadeStatus.V7_FALLBACK,
            "v7_only_candidate",
            policy.policy_sha256,
            provider_status,
            None,
            None,
            {},
        )

    selected = selection.selected
    selected_corners = selected.corners
    v7_corners = _corners(v7_candidate)
    mask_corners = _corners(mask_candidate)
    v7_distance = _distance(selected_corners, v7_corners, image_size)
    mask_distance = _distance(selected_corners, mask_corners, image_size)
    v7_mask_distance = _distance(v7_corners, mask_corners, image_size)
    mask_strong = _strong_mask(mask_candidate, policy)
    selector_config = policy.parameters.selector_config
    cluster_seed_support = (
        selection.selected_edge_cluster_seed_support_count
        or selected.seed_support_count
    )
    cluster_source_group_support = (
        selection.selected_edge_cluster_source_group_count
        or selected.seed_source_group_count
    )
    v7_agrees = (
        v7_distance is not None
        and v7_distance <= float(selector_config.v7_edge_agreement_distance)
    )
    strong_mask_agrees = (
        mask_strong
        and mask_distance is not None
        and mask_distance <= float(selector_config.edge_mask_agreement_distance)
    )
    evidence = {
        "selection_reason": selection.reason,
        "selected_valid_side_count": selected.evidence.valid_side_count,
        "selected_exterior_score": selected.evidence.score,
        "selected_area_ratio": selected.area_ratio,
        "selected_prior_score_normalized": selected.prior_score_normalized,
        "selected_seed_support_count": selected.seed_support_count,
        "selected_seed_source_group_count": selected.seed_source_group_count,
        "selected_edge_cluster_size": selection.selected_edge_cluster_size,
        "selected_edge_cluster_seed_support_count": cluster_seed_support,
        "selected_edge_cluster_source_group_count": cluster_source_group_support,
        "competing_edge_candidate_id": selection.competing_edge_candidate_id,
        "competing_edge_distance": selection.competing_edge_distance,
        "edge_score_margin": selection.edge_score_margin,
        "selected_v7_distance": v7_distance,
        "selected_mask_distance": mask_distance,
        "v7_mask_distance": v7_mask_distance,
        "mask_strong": mask_strong,
        "v7_agrees": v7_agrees,
        "strong_mask_agrees": strong_mask_agrees,
    }
    selected_mask = (
        mask_candidate is not None
        and selected.candidate_id == mask_candidate.get("candidate_id")
    )
    mask_dependent_selection = selected_mask or selection.reason in {
        "edge_mask_consensus_rescue",
        "v7_mask_consensus_rescue",
    }
    if mask_dependent_selection and not mask_strong:
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "weak_mask_evidence",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    if selection.reason == "mask_only":
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "mask_only_unconfirmed",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    if (
        selected.evidence.valid_side_count
        < selector_config.minimum_automatic_valid_sides
        and not v7_agrees
        and not strong_mask_agrees
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "insufficient_scanner_exterior_evidence",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    conflict = float(selector_config.manual_conflict_distance)
    if (
        selection.reason == "v7_edge_agreement_rescue"
        and mask_strong
        and mask_distance is not None
        and mask_distance >= conflict
        and selection.edge_score_margin is not None
        and selection.edge_score_margin < 0.0
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "unstable_v7_edge_rescue",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    if (
        mask_strong
        and v7_distance is not None
        and v7_distance >= conflict
        and mask_distance is not None
        and mask_distance >= conflict
        and v7_mask_distance is not None
        and v7_mask_distance >= conflict
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "strong_three_way_conflict",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    selected_v7 = (
        v7_candidate is not None
        and selected.candidate_id == v7_candidate.get("candidate_id")
    )
    selected_edge = not selected_v7 and not selected_mask
    edge_supported = (
        cluster_seed_support >= selector_config.minimum_edge_seed_support
        and cluster_source_group_support
        >= selector_config.minimum_edge_source_group_support
    )
    if (
        selected_edge
        and v7_distance is not None
        and v7_distance >= conflict
        and not strong_mask_agrees
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "v7_edge_unresolved_conflict",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    if (
        selected_edge
        and not edge_supported
        and not v7_agrees
        and not strong_mask_agrees
        and selected.evidence.score
        < float(selector_config.minimum_unopposed_exterior_score)
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "unsupported_single_source_selection",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    if (
        selected_edge
        and not v7_agrees
        and not strong_mask_agrees
        and selected.evidence.score
        < float(selector_config.minimum_unopposed_exterior_score)
        and (
            selection.edge_score_margin is None
            or selection.edge_score_margin
            < 4.0 * float(selector_config.minimum_automatic_score_margin)
        )
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "weak_unopposed_edge_evidence",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    if (
        selected_edge
        and not v7_agrees
        and not strong_mask_agrees
        and selection.competing_edge_distance is not None
        and selection.competing_edge_distance >= conflict
        and selection.edge_score_margin is not None
        and selection.edge_score_margin
        < float(selector_config.minimum_automatic_score_margin)
    ):
        return V8CascadeDecision(
            V8CascadeStatus.MANUAL_REVIEW,
            "unstable_edge_ranking",
            policy.policy_sha256,
            provider_status,
            selected.candidate_id,
            selected_corners,
            evidence,
        )
    return V8CascadeDecision(
        V8CascadeStatus.AUTOMATIC,
        selection.reason,
        policy.policy_sha256,
        provider_status,
        selected.candidate_id,
        selected_corners,
        evidence,
    )


__all__ = [
    "V8CascadeDecision",
    "V8CascadeStatus",
    "V8DormantResult",
    "decide_v8_cascade",
]
