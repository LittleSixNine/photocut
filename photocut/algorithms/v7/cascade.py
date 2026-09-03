"""Bounded, deterministic arbitration between one V7 result and one v5.2 result."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .geometry import GeometryError, polygon_iou, validate_quad


SAFE_THRESHOLD = 0.78
AGGRESSIVE_THRESHOLD = 0.62
AGREEMENT_DISTANCE = 0.02
AGREEMENT_IOU = 0.90
LEGACY_MEAN_CONFIDENCE = 0.70
LEGACY_MIN_CONFIDENCE = 0.50
SCANNER_BOUNDARY_MAX_DROP = 0.08
SCANNER_BOUNDARY_SWITCH_MARGIN = 0.08

_TERMINAL_STATUSES = {"no_primary_photo", "cancelled"}
_HARD_RISKS = {
    "multiple_primary_ambiguity", "suspected_outer_frame", "geometry_invalid",
    "unrecoverable_error", "refinement_cancelled", "refinement_timeout",
}

CASCADE_POLICY_VERSION = "v7-auto-v3"
_POLICY_PAYLOAD = {
    "version": CASCADE_POLICY_VERSION,
    "safe_threshold": SAFE_THRESHOLD,
    "aggressive_threshold": AGGRESSIVE_THRESHOLD,
    "agreement_distance": AGREEMENT_DISTANCE,
    "agreement_iou": AGREEMENT_IOU,
    "legacy_mean_confidence": LEGACY_MEAN_CONFIDENCE,
    "legacy_min_confidence": LEGACY_MIN_CONFIDENCE,
    "max_v7_calls": 1,
    "max_legacy_calls": 1,
    "v7_cross_check_candidates": 5,
    "scanner_white_supplement_candidates": 3,
    "scanner_boundary_rerank": True,
    "scanner_boundary_max_drop": SCANNER_BOUNDARY_MAX_DROP,
    "scanner_boundary_switch_margin": SCANNER_BOUNDARY_SWITCH_MARGIN,
    "scanner_legacy_takeover": False,
    "total_deadline_ms": 1500,
}
CASCADE_POLICY_SHA256 = hashlib.sha256(json.dumps(
    _POLICY_PAYLOAD, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CascadeDecision:
    action: str
    detector_used: str
    needs_legacy: bool = False
    reasons: tuple[str, ...] = ()
    selected: Mapping[str, Any] | None = None


def _status(info: Mapping[str, Any]) -> str:
    return str(info.get("detection_status", info.get("status", ""))).lower()


def _corners(info: Mapping[str, Any]) -> Any:
    return info.get("corners") or info.get("boundary_corners") or info.get("algorithm_corners")


def _confidence(info: Mapping[str, Any]) -> float | None:
    value = info.get("overall_confidence")
    if value is None:
        values = info.get("confidences") or info.get("corner_confidences") or ()
        try:
            values = tuple(float(item) for item in values)
        except (TypeError, ValueError):
            return None
        if not values:
            return None
        value = sum(values) / len(values)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def _risks(info: Mapping[str, Any]) -> set[str]:
    values = info.get("risks") or ()
    return {str(item) for item in values if item not in (None, "none")}


def _valid_quad(info: Mapping[str, Any], image_size: Sequence[float]) -> tuple[tuple[float, float], ...] | None:
    corners = _corners(info)
    if corners is None:
        return None
    try:
        return validate_quad(corners, image_size=image_size)
    except (GeometryError, TypeError, ValueError):
        return None


def _normalise_quad(quad: Sequence[Sequence[float]], image_size: Sequence[float]) -> tuple[tuple[float, float], ...]:
    width, height = float(image_size[0]), float(image_size[1])
    if width < 2 or height < 2:
        raise ValueError("image_size must be at least 2x2")
    return tuple((float(x) / (width - 1.0), float(y) / (height - 1.0)) for x, y in quad)


def _agreement(v7_quad: Sequence[Sequence[float]], legacy_quad: Sequence[Sequence[float]], image_size: Sequence[float]) -> bool:
    a = _normalise_quad(v7_quad, image_size)
    b = _normalise_quad(legacy_quad, image_size)
    distance = sum(math.hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(a, b)) / 4.0
    iou = float(polygon_iou(a, b))
    return distance <= AGREEMENT_DISTANCE or iou >= AGREEMENT_IOU


def _legacy_strong(info: Mapping[str, Any], image_size: Sequence[float]) -> bool:
    if not bool(info.get("success", True)) or _valid_quad(info, image_size) is None:
        return False
    values = info.get("confidences") or info.get("corner_confidences")
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        return False
    try:
        scores = tuple(float(item) for item in values)
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(item) and 0.0 <= item <= 1.0 for item in scores) and (
        sum(scores) / 4.0 >= LEGACY_MEAN_CONFIDENCE and min(scores) >= LEGACY_MIN_CONFIDENCE
    )


def _audit_score(audit: Mapping[str, Any]) -> float:
    scores = audit.get("stage_scores")
    if not isinstance(scores, Mapping):
        return 0.0
    for key in ("full_score", "score", "pre_score"):
        value = scores.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return max(0.0, min(1.0, value))
    return 0.0


def _audit_boundary_score(audit: Mapping[str, Any]) -> float | None:
    scores = audit.get("stage_scores")
    components = scores.get("components") if isinstance(scores, Mapping) else None
    value = components.get("scanner_boundary_score") if isinstance(components, Mapping) else None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value)) if math.isfinite(value) else None


def _audit_rank(audit: Mapping[str, Any]) -> int:
    ranks = audit.get("stage_ranks")
    value = ranks.get("selected") if isinstance(ranks, Mapping) else None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return 1_000_000
    return value if value > 0 else 1_000_000


def _scanner_supplement(audit: Mapping[str, Any]) -> bool:
    sources = audit.get("sources")
    if isinstance(sources, str):
        sources = (sources,)
    try:
        source_set = {str(value) for value in sources or ()}
    except TypeError:
        source_set = set()
    evidence = audit.get("pre_truncation_risk_evidence")
    return (
        source_set == {"background:border_connected"}
        and isinstance(evidence, Mapping)
        and evidence.get("supplemental_only") is True
    )


def _ranked_topk(v7: Mapping[str, Any], image_size: Sequence[float]) -> tuple[tuple[Mapping[str, Any], tuple[tuple[float, float], ...]], ...]:
    audits = v7.get("candidate_audit")
    if not isinstance(audits, (tuple, list)):
        return ()
    ranked = []
    for audit in audits:
        if not isinstance(audit, Mapping) or audit.get("truncation_stage") != "selected":
            continue
        corners = None
        for key in ("adopted_refined_corners", "pre_topk_corners", "original_legal_corners"):
            if audit.get(key) is not None:
                corners = audit.get(key)
                break
        try:
            quad = validate_quad(corners, image_size=image_size)
        except (GeometryError, TypeError, ValueError):
            continue
        ranked.append((audit, quad))
    ranked.sort(key=lambda entry: (
        -_audit_score(entry[0]), _audit_rank(entry[0]),
        str(entry[0].get("candidate_id", "")),
    ))
    regular = [entry for entry in ranked if not _scanner_supplement(entry[0])]
    supplements = [entry for entry in ranked if _scanner_supplement(entry[0])]
    return tuple(regular[:5] + supplements[:3])


def _same_quad(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]],
               image_size: Sequence[float]) -> bool:
    diagonal = math.hypot(float(image_size[0]), float(image_size[1]))
    if diagonal <= 0:
        return False
    return max(math.hypot(float(ax) - float(bx), float(ay) - float(by))
               for (ax, ay), (bx, by) in zip(a, b)) / diagonal <= 1e-6


def _scanner_agreement_candidates(
    v7: Mapping[str, Any], primary: Sequence[Sequence[float]] | None,
    legacy: Sequence[Sequence[float]], image_size: Sequence[float],
) -> tuple[tuple[Mapping[str, Any] | None, tuple[tuple[float, float], ...], float], ...]:
    matches = []
    primary_seen = False
    for audit, quad in _ranked_topk(v7, image_size):
        if primary is not None and _same_quad(quad, primary, image_size):
            primary_seen = True
        if not _agreement(quad, legacy, image_size):
            continue
        boundary = _audit_boundary_score(audit)
        matches.append((audit, quad, -1.0 if boundary is None else boundary))
    if primary is not None and not primary_seen and _agreement(primary, legacy, image_size):
        try:
            boundary = float(v7.get("scanner_boundary_score", -1.0))
        except (TypeError, ValueError):
            boundary = -1.0
        matches.append((None, tuple(tuple(point) for point in primary), boundary))
    matches.sort(key=lambda entry: (
        -entry[2],
        -(_audit_score(entry[0]) if entry[0] is not None else _confidence(v7) or 0.0),
        _audit_rank(entry[0]) if entry[0] is not None else 0,
        str(entry[0].get("candidate_id", "")) if entry[0] is not None else "",
    ))
    return tuple(matches)


def _primary_boundary_score(
    v7: Mapping[str, Any], primary: Sequence[Sequence[float]] | None,
    image_size: Sequence[float],
) -> float | None:
    if primary is None:
        return None
    for audit, candidate_quad in _ranked_topk(v7, image_size):
        if _same_quad(candidate_quad, primary, image_size):
            return _audit_boundary_score(audit)
    return None


def _select_v7_candidate(v7: Mapping[str, Any], audit: Mapping[str, Any],
                         quad: Sequence[Sequence[float]], image_size: Sequence[float],
                         rank: int) -> Mapping[str, Any]:
    selected = dict(v7)
    corners = [[float(x), float(y)] for x, y in quad]
    for key in ("corners", "boundary_corners", "algorithm_boundary_corners", "algorithm_corners"):
        selected[key] = [point[:] for point in corners]
    preview_size = selected.get("preview_size")
    try:
        preview_width, preview_height = float(preview_size[0]), float(preview_size[1])
        width, height = float(image_size[0]), float(image_size[1])
        preview = [[int(round(x * preview_width / width)), int(round(y * preview_height / height))]
                   for x, y in quad]
    except (TypeError, ValueError, IndexError, ZeroDivisionError):
        preview = []
    selected["algorithm_preview_corners"] = [point[:] for point in preview]
    selected["preview_corners"] = [point[:] for point in preview]
    sources = audit.get("sources")
    if isinstance(sources, (tuple, list)):
        selected["candidate_sources"] = [str(value) for value in sources]
    score = _audit_score(audit)
    selected["overall_confidence"] = score
    selected["cascade_v7_candidate_id"] = str(audit.get("candidate_id", ""))
    selected["cascade_v7_candidate_rank"] = int(rank)
    selected["cascade_v7_candidate_score"] = score
    return selected


def decide_auto(v7: Mapping[str, Any], *, image_size: Sequence[float], legacy: Mapping[str, Any] | None = None) -> CascadeDecision:
    """Choose V7, request one legacy check, or require manual review."""
    if not isinstance(v7, Mapping):
        return CascadeDecision("manual_review", "manual_review", reasons=("invalid_v7_result",))
    status = _status(v7)
    if status in _TERMINAL_STATUSES or status == "no_photo_evidence":
        return CascadeDecision("manual_review", "manual_review", reasons=(status or "terminal_status",))
    quad = _valid_quad(v7, image_size)
    confidence = _confidence(v7)
    risks = _risks(v7)
    hard = risks & _HARD_RISKS
    if quad is not None and confidence is not None and confidence >= SAFE_THRESHOLD and not hard:
        return CascadeDecision("accept_v7", "v7", selected=v7)
    if legacy is None:
        return CascadeDecision("run_v5_2", "pending", needs_legacy=True,
                                reasons=tuple(sorted(hard or {"v7_low_confidence"})))
    legacy_quad = _valid_quad(legacy, image_size)
    scanner_white = str(v7.get("scene_profile", "generic_single")) == "scanner_white"
    if scanner_white and legacy_quad is not None:
        matches = _scanner_agreement_candidates(v7, quad, legacy_quad, image_size)
        if matches and matches[0][2] >= 0.0:
            audit, candidate_quad, boundary_score = matches[0]
            if audit is None or quad is not None and _same_quad(candidate_quad, quad, image_size):
                return CascadeDecision(
                    "accept_v7", "v7", reasons=("cross_algorithm_agreement",), selected=v7,
                )
            primary_boundary = _primary_boundary_score(v7, quad, image_size)
            if (quad is not None and _agreement(quad, legacy_quad, image_size) and
                    primary_boundary is not None and
                    boundary_score < primary_boundary + SCANNER_BOUNDARY_SWITCH_MARGIN):
                return CascadeDecision(
                    "accept_v7", "v7", reasons=("cross_algorithm_agreement",), selected=v7,
                )
            reported_rank = _audit_rank(audit)
            if reported_rank == 1_000_000:
                reported_rank = 1
            selected = _select_v7_candidate(
                v7, audit, candidate_quad, image_size, reported_rank
            )
            if not _scanner_supplement(audit):
                if (primary_boundary is not None and
                        boundary_score + SCANNER_BOUNDARY_MAX_DROP < primary_boundary):
                    return CascadeDecision(
                        "manual_review", "manual_review",
                        reasons=("scanner_boundary_conflict",), selected=None,
                    )
            return CascadeDecision(
                "accept_v7", "v7", reasons=((
                    "cross_algorithm_agreement_supplement"
                    if _scanner_supplement(audit)
                    else "cross_algorithm_agreement_topk"
                ),), selected=selected,
            )
    if quad is not None and legacy_quad is not None and _agreement(quad, legacy_quad, image_size):
        return CascadeDecision("accept_v7", "v7", reasons=("cross_algorithm_agreement",), selected=v7)
    if not scanner_white and legacy_quad is not None:
        for rank, (audit, candidate_quad) in enumerate(_ranked_topk(v7, image_size), 1):
            if _agreement(candidate_quad, legacy_quad, image_size):
                reported_rank = _audit_rank(audit)
                if reported_rank == 1_000_000:
                    reported_rank = rank
                selected = _select_v7_candidate(
                    v7, audit, candidate_quad, image_size, reported_rank
                )
                return CascadeDecision(
                    "accept_v7", "v7", reasons=((
                        "cross_algorithm_agreement_supplement"
                        if _scanner_supplement(audit)
                        else "cross_algorithm_agreement_topk"
                    ),),
                    selected=selected,
                )
    if (not scanner_white and _legacy_strong(legacy, image_size) and
            (confidence is None or confidence < AGGRESSIVE_THRESHOLD or hard)):
        return CascadeDecision("accept_v5_2", "v5.2", reasons=("legacy_strong_evidence",), selected=legacy)
    return CascadeDecision("manual_review", "manual_review", reasons=("cross_algorithm_conflict",), selected=None)


__all__ = ["CASCADE_POLICY_SHA256", "CASCADE_POLICY_VERSION", "CascadeDecision", "decide_auto"]
