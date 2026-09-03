"""Pure V8 candidate-oracle metrics and released-development gate.

This module intentionally does not select a production result.  It measures
whether a strict candidate exists in the fixed V7 pool plus V8 shadow
sources, and keeps full-pool feasibility separate from bounded candidate
budgets.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from photocut.algorithms.v7.evaluation import corner_errors


STRICT_MAX_NORMALIZED_CORNER_ERROR = 0.005
DEFAULT_EDGE_BUDGETS = (8, 16, 32, 64, 128, 256)


def _strict(corners: Any, truth: Any, image_size: Sequence[int]) -> bool:
    result = corner_errors(corners, truth, image_size=image_size)
    return bool(result.get("valid")) and max(
        result.get("normalized_per_corner", (1.0,)), default=1.0
    ) <= STRICT_MAX_NORMALIZED_CORNER_ERROR


def first_strict_rank(
    candidates: Iterable[Any],
    truth: Any,
    *,
    image_size: Sequence[int],
) -> int | None:
    """Return the one-based rank of the first strict candidate, if present."""
    for rank, candidate in enumerate(candidates, 1):
        corners = getattr(candidate, "corners", candidate)
        if _strict(corners, truth, image_size):
            return rank
    return None


def _count_rows(rows: Sequence[Mapping[str, Any]], budgets: tuple[int, ...]) -> dict[str, Any]:
    v7_count = 0
    edge_count = 0
    mask_count = 0
    union_count = 0
    edge_unique = 0
    mask_unique = 0
    budget_counts = {str(value): 0 for value in budgets}
    for row in rows:
        v7 = row.get("v7_strict") is True
        edge_rank = row.get("edge_first_strict_rank")
        if edge_rank is not None and (type(edge_rank) is not int or edge_rank < 1):
            raise ValueError("edge_first_strict_rank must be a positive integer or null")
        edge = edge_rank is not None
        mask = row.get("mask_strict") is True
        v7_count += int(v7)
        edge_count += int(edge)
        mask_count += int(mask)
        union_count += int(v7 or edge or mask)
        edge_unique += int(edge and not v7)
        mask_unique += int(mask and not v7 and not edge)
        for budget in budgets:
            budget_counts[str(budget)] += int(v7 or mask or (edge_rank is not None and edge_rank <= budget))
    count = len(rows)
    return {
        "sample_count": count,
        "v7_oracle_count": v7_count,
        "edge_oracle_count": edge_count,
        "mask_oracle_count": mask_count,
        "union_oracle_count": union_count,
        "edge_unique_vs_v7_count": edge_unique,
        "mask_unique_vs_v7_edge_count": mask_unique,
        "v8_net_gain_vs_v7_count": union_count - v7_count,
        "union_oracle_rate": union_count / count if count else 0.0,
        "edge_budget_union_counts": budget_counts,
    }


def summarize_candidate_oracle(
    rows: Iterable[Mapping[str, Any]],
    *,
    edge_budgets: Sequence[int] = DEFAULT_EDGE_BUDGETS,
) -> dict[str, Any]:
    """Summarize full and capped V7+edge+mask candidate availability."""
    materialized = tuple(rows)
    budgets = tuple(edge_budgets)
    if not budgets or any(type(value) is not int or value < 1 for value in budgets):
        raise ValueError("edge budgets must be positive integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("edge budgets must be unique")
    for row in materialized:
        if row.get("training_subset") not in {"fit", "calibration"}:
            raise ValueError("training_subset must be fit or calibration")
    summary = _count_rows(materialized, budgets)
    summary["strict_max_normalized_corner_error"] = STRICT_MAX_NORMALIZED_CORNER_ERROR
    summary["edge_budgets"] = list(budgets)
    summary["subsets"] = {
        subset: _count_rows(
            tuple(row for row in materialized if row["training_subset"] == subset),
            budgets,
        )
        for subset in ("fit", "calibration")
    }
    return summary


def candidate_feasibility_gate(
    summary: Mapping[str, Any],
    *,
    frozen_accessed: bool,
    source_modified_count: int,
    minimum_rate: float = 0.95,
    minimum_net_gain: int = 30,
) -> dict[str, Any]:
    """Apply the predeclared released-development candidate-only gate."""
    sample_count = int(summary.get("sample_count", 0))
    calibration = summary.get("subsets", {}).get("calibration", {})
    calibration_count = int(calibration.get("sample_count", 0))
    required = math.ceil(minimum_rate * sample_count)
    calibration_required = math.ceil(minimum_rate * calibration_count)
    conditions = {
        "population_union_at_least_95_percent": int(summary.get("union_oracle_count", 0)) >= required,
        "population_net_gain_at_least_30": int(summary.get("v8_net_gain_vs_v7_count", 0)) >= minimum_net_gain,
        "calibration_union_at_least_95_percent": int(calibration.get("union_oracle_count", 0)) >= calibration_required,
        "calibration_net_gain_positive": int(calibration.get("v8_net_gain_vs_v7_count", 0)) > 0,
        "frozen_not_accessed": frozen_accessed is False,
        "source_unchanged": type(source_modified_count) is int and source_modified_count == 0,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    if frozen_accessed:
        failed.append("frozen_accessed")
    return {
        "passed": not failed,
        "research_only": True,
        "production_promotion": False,
        "required_union_count": required,
        "required_calibration_union_count": calibration_required,
        "minimum_rate": minimum_rate,
        "minimum_net_gain": minimum_net_gain,
        "conditions": conditions,
        "failed_conditions": failed,
    }


def summarize_end_to_end(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize real policy terminal states over the full eligible population."""
    materialized = tuple(rows)
    allowed = {"automatic", "manual_review", "v7_fallback", "cancelled", "error"}
    counts = {status: 0 for status in allowed}
    strict_count = 0
    catastrophic_count = 0
    automatic_count = 0
    for row in materialized:
        status = row.get("decision_status")
        if status not in allowed:
            raise ValueError("invalid V8 decision_status")
        strict = row.get("automatic_strict")
        catastrophic = row.get("automatic_catastrophic")
        if type(strict) is not bool or type(catastrophic) is not bool:
            raise ValueError("automatic outcome flags must be booleans")
        if status in {"manual_review", "cancelled", "error"} and (strict or catastrophic):
            raise ValueError("non-automatic terminal state cannot carry automatic outcome")
        counts[status] += 1
        if status in {"automatic", "v7_fallback"}:
            automatic_count += 1
            strict_count += int(strict)
            catastrophic_count += int(catastrophic)
    sample_count = len(materialized)
    manual_count = counts["manual_review"] + counts["cancelled"] + counts["error"]
    return {
        "sample_count": sample_count,
        "automatic_count": automatic_count,
        "automatic_strict_count": strict_count,
        "automatic_strict_rate": strict_count / sample_count if sample_count else 0.0,
        "automatic_catastrophic_count": catastrophic_count,
        "manual_or_rescan_count": manual_count,
        "manual_or_rescan_rate": manual_count / sample_count if sample_count else 0.0,
        "v7_fallback_count": counts["v7_fallback"],
        "decision_status_counts": {
            status: count for status, count in sorted(counts.items()) if count
        },
    }


def end_to_end_gate(
    summary: Mapping[str, Any],
    *,
    frozen_accessed: bool,
    source_modified_count: int,
    minimum_automatic_strict_rate: float = 0.95,
    maximum_manual_rate: float = 0.05,
) -> dict[str, Any]:
    """Apply the predeclared V8 policy gate to full-population outcomes."""
    sample_count = int(summary.get("sample_count", 0))
    strict_required = math.ceil(minimum_automatic_strict_rate * sample_count)
    manual_allowed = math.floor(maximum_manual_rate * sample_count)
    conditions = {
        "automatic_strict_at_least_95_percent": int(
            summary.get("automatic_strict_count", 0)
        ) >= strict_required,
        "automatic_catastrophic_zero": int(
            summary.get("automatic_catastrophic_count", 0)
        ) == 0,
        "manual_or_rescan_at_most_5_percent": int(
            summary.get("manual_or_rescan_count", 0)
        ) <= manual_allowed,
        "frozen_not_accessed": frozen_accessed is False,
        "source_unchanged": type(source_modified_count) is int and source_modified_count == 0,
    }
    failed = [name for name, passed in conditions.items() if not passed]
    return {
        "passed": not failed,
        "research_only": True,
        "production_promotion": False,
        "minimum_automatic_strict_rate": minimum_automatic_strict_rate,
        "maximum_manual_rate": maximum_manual_rate,
        "required_automatic_strict_count": strict_required,
        "allowed_manual_or_rescan_count": manual_allowed,
        "conditions": conditions,
        "failed_conditions": failed,
    }


__all__ = [
    "DEFAULT_EDGE_BUDGETS",
    "STRICT_MAX_NORMALIZED_CORNER_ERROR",
    "candidate_feasibility_gate",
    "end_to_end_gate",
    "first_strict_rank",
    "summarize_end_to_end",
    "summarize_candidate_oracle",
]
