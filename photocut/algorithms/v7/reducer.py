"""Deterministic, bounded candidate fusion and post-audit Top-K selection.

The reducer is intentionally independent of image features and scoring.  Providers
emit ordinary mappings; this module validates geometry, fuses nearby quads and
keeps enough provenance for later audit/evaluation.  ``select_topk`` is a separate
post-audit operation and therefore never runs during provider fusion.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .geometry import GeometryError, min_normalized_edge, normalized_corner_distance, order_quad, quad_area_ratio, validate_quad


@dataclass(frozen=True)
class RejectedCandidate:
    candidate_id: str
    reason: str
    provider: str | None = None


@dataclass(frozen=True)
class FusedCandidates:
    """Output of :func:`fuse_candidates`.

    ``candidates`` are canonical mappings (and are never Top-K truncated).  The
    convenience properties keep callers from depending on the internal rejected
    record representation.
    """

    candidates: tuple[Mapping[str, Any], ...] = ()
    rejected: tuple[RejectedCandidate, ...] = ()
    fused_limit: int = 40

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "rejected", tuple(self.rejected))

    @property
    def rejected_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.rejected)

    @property
    def rejected_reasons(self) -> dict[str, Any]:
        # Keep a stable short key for the common case.  If malformed/duplicate
        # IDs recur across providers, retain every reason as a tuple instead of
        # silently overwriting an audit record.
        result: dict[str, Any] = {}
        for item in self.rejected:
            if item.candidate_id not in result:
                result[item.candidate_id] = item.reason
            elif isinstance(result[item.candidate_id], tuple):
                result[item.candidate_id] = result[item.candidate_id] + (item.reason,)
            else:
                result[item.candidate_id] = (result[item.candidate_id], item.reason)
        return result

    @property
    def accepted(self) -> tuple[Mapping[str, Any], ...]:
        return self.candidates

    @property
    def fused(self) -> tuple[Mapping[str, Any], ...]:
        return self.candidates

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)

    def __getitem__(self, index):
        return self.candidates[index]

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [dict(item) for item in self.candidates],
                "rejected_ids": list(self.rejected_ids),
                "rejected_reasons": dict(self.rejected_reasons),
                "fused_limit": self.fused_limit}


@dataclass(frozen=True)
class TopKSelection:
    candidates: tuple[Mapping[str, Any], ...] = ()
    truncation_trace: tuple[Mapping[str, Any], ...] = ()

    @property
    def selected(self) -> tuple[Mapping[str, Any], ...]:
        return self.candidates

    @property
    def trace(self) -> tuple[Mapping[str, Any], ...]:
        return self.truncation_trace

    @property
    def audit_trace(self) -> tuple[Mapping[str, Any], ...]:
        return self.truncation_trace

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self):
        # Supports the ergonomic ``selected, trace = select_topk(...)`` form
        # while retaining named fields for production callers.
        yield self.candidates
        yield self.truncation_trace

    def __getitem__(self, index):
        return (self.candidates, self.truncation_trace)[index]

    def to_dict(self) -> dict[str, Any]:
        return {"candidates": [dict(item) for item in self.candidates],
                "truncation_trace": [dict(item) for item in self.truncation_trace]}


def _param(params: Any, name: str, default: Any) -> Any:
    return getattr(params, name, default) if params is not None else default


def _provider_groups(provider_results: Any) -> tuple[dict[str, list[Mapping[str, Any]]], list[RejectedCandidate]]:
    """Normalize ProviderResult, ``(provider, candidates)`` and mapping inputs."""
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    rejected: list[RejectedCandidate] = []
    def add_result(provider: str, values: Any, status: Any = None) -> None:
        status_value = getattr(status, "value", status)
        if isinstance(values, Mapping): values = [values]
        try: iterable = list(values or ())
        except TypeError: iterable = [values]
        if status_value is not None and (str(status_value) not in ("success", "no_candidate") or
                                         str(status_value) == "no_candidate" and iterable):
            for index, value in enumerate(iterable):
                rejected.append(RejectedCandidate(_candidate_id(value, index),
                                                  f"provider_status:{status_value}", provider))
            return
        groups[provider].extend(iterable)
    if provider_results is None:
        return groups, rejected
    if isinstance(provider_results, Mapping):
        # A single candidate mapping is accepted as a one-provider input; a
        # provider->candidate-list mapping is the more common form.
        if "candidates" in provider_results and ("provider" in provider_results or "status" in provider_results):
            provider = str(provider_results.get("provider") or provider_results.get("name") or "unknown")
            add_result(provider, provider_results.get("candidates"), provider_results.get("status"))
            return groups, rejected
        if "corners" in provider_results or "candidate_id" in provider_results or "id" in provider_results:
            provider = str(provider_results.get("provider") or provider_results.get("source") or "unknown")
            groups[provider].append(provider_results)
            return groups, rejected
        for provider, values in provider_results.items():
            if hasattr(values, "candidates"):
                add_result(str(getattr(values, "provider", provider)), values.candidates, getattr(values, "status", None)); continue
            if isinstance(values, Mapping) and "candidates" in values:
                add_result(str(values.get("provider", provider)), values.get("candidates"), values.get("status")); continue
            add_result(str(provider), values)
        return groups, rejected
    try:
        items = list(provider_results)
    except TypeError:
        items = [provider_results]
    for item in items:
        if hasattr(item, "provider") and hasattr(item, "candidates"):
            provider, values = str(item.provider), item.candidates
            add_result(provider, values, getattr(item, "status", None))
        elif isinstance(item, Mapping) and "candidates" in item and ("provider" in item or "name" in item):
            provider = str(item.get("provider") or item.get("name"))
            values = item.get("candidates") or ()
            add_result(provider, values, item.get("status"))
        elif isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            values = item[1]
            add_result(item[0], values)
        elif isinstance(item, Mapping):
            provider = str(item.get("provider") or item.get("source") or "unknown")
            groups[provider].append(item)
    return groups, rejected


def _candidate_id(item: Any, index: int) -> str:
    if isinstance(item, Mapping):
        value = item.get("candidate_id", item.get("id"))
        if isinstance(value, str) and value.strip():
            return value
    return f"<malformed:{index:04d}>"


def _sources(item: Mapping[str, Any], provider: str) -> tuple[str, ...]:
    values = item.get("sources")
    if values is None:
        values = (item.get("source"),)
    if isinstance(values, str):
        values = (values,)
    try:
        out = tuple(sorted({str(value) for value in values if value not in (None, "")}))
    except TypeError:
        out = ()
    return out or (provider,)


def _validate_candidate(item: Any, provider: str, image_size: Sequence[float] | None,
                        *, min_area_ratio: float, max_area_ratio: float,
                        min_edge_ratio: float, max_edge_ratio: float) -> tuple[dict[str, Any], str | None]:
    cid = _candidate_id(item, 0)
    if not isinstance(item, Mapping):
        return {"candidate_id": cid}, "malformed_candidate"
    raw_id = item.get("candidate_id", item.get("id"))
    if not isinstance(raw_id, str) or not raw_id.strip():
        return {"candidate_id": cid}, "missing_candidate_id"
    for score_key in ("score", "pre_score", "pre_score_total", "cheap_score"):
        if score_key in item and item[score_key] is not None:
            try:
                if not math.isfinite(float(item[score_key])):
                    return {"candidate_id": cid}, "non_finite_score"
            except (TypeError, ValueError):
                return {"candidate_id": cid}, "malformed_score"
    corners = item.get("corners", item.get("original_legal_corners"))
    if corners is None:
        return {"candidate_id": cid}, "missing_corners"
    try:
        # Check exact duplicates before segment-intersection checks: repeated
        # vertices are a distinct and more useful rejection reason than the
        # overlapping-edge symptom produced by a generic polygon validator.
        raw_points = corners.tolist() if hasattr(corners, "tolist") else corners
        if isinstance(raw_points, Sequence) and len(raw_points) == 4:
            normalized_points = [tuple(point.tolist() if hasattr(point, "tolist") else point) for point in raw_points]
            if len({tuple(point) for point in normalized_points}) != 4:
                raise GeometryError("quadrilateral contains duplicate points")
        # validate_quad uses the image convention and rejects duplicate,
        # concave, bow-tie and non-finite coordinates without repairing them.
        quad = validate_quad(corners, image_size, min_area_ratio=min_area_ratio,
                             max_area_ratio=max_area_ratio, min_edge_ratio=min_edge_ratio)
        if image_size is not None:
            diagonal = math.hypot(float(image_size[0]), float(image_size[1]))
            lengths = [math.hypot(quad[(i + 1) % 4][0] - quad[i][0],
                                   quad[(i + 1) % 4][1] - quad[i][1]) for i in range(4)]
            if max(lengths) / diagonal > max_edge_ratio:
                raise GeometryError("quadrilateral has a too-long normalized edge")
    except (GeometryError, TypeError, ValueError, OverflowError) as exc:
        message = str(exc).lower()
        if "finite" in message or "nan" in message or "infinite" in message:
            reason = "non_finite_geometry"
        elif "duplicate" in message:
            reason = "duplicate_points"
        elif "self-intersect" in message:
            reason = "self_intersecting"
        elif "concave" in message or "convex" in message:
            reason = "non_convex_geometry"
        elif "area" in message or "small" in message:
            reason = "area_out_of_bounds"
        elif "edge" in message:
            reason = "edge_length_out_of_bounds"
        else:
            reason = "invalid_geometry"
        return {"candidate_id": cid}, reason
    result = dict(item)
    result["candidate_id"] = cid
    result["id"] = str(result.get("id") or cid)
    result["corners"] = quad
    result["provider"] = provider
    result["providers"] = (provider,)
    result["sources"] = _sources(item, provider)
    result["provenance"] = ({"candidate_id": cid, "provider": provider,
                              "sources": result["sources"]},)
    return result, None


def _sort_key(item: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(item, Mapping):
        # Keep malformed provider output sortable and deterministic; it is
        # passed unchanged to _validate_candidate so the rejection is audited.
        return ("", type(item).__name__, repr(item))
    return (str(item.get("candidate_id", "")), str(item.get("source", "")), str(item.get("id", "")))


def fuse_candidates(provider_results: Any, image_size: Sequence[float] | None = None,
                    params: Any = None, *, fused_limit: int | None = None,
                    per_source_quota: int | None = None,
                    dedup_distance: float | None = None,
                    source_quotas: Mapping[str, int] | None = None,
                    limit: int | None = None) -> FusedCandidates:
    """Validate, deduplicate and fairly fuse provider output.

    Provider queues are sorted by stable IDs and consumed round-robin.  A default
    hard quota of ``ceil(fused_limit / provider_count)`` prevents one provider
    from flooding the fused pool; a caller may override it explicitly.
    """
    fused_limit_value = fused_limit if fused_limit is not None else limit
    bound = int(fused_limit_value if fused_limit_value is not None else _param(params, "fused_budget", 40))
    if bound <= 0:
        raise ValueError("fused_limit must be positive")
    if bound > 40:
        raise ValueError("fused_limit cannot exceed hard limit 40")
    distance = float(dedup_distance if dedup_distance is not None else _param(params, "dedup_distance", 0.015))
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("dedup_distance must be finite and non-negative")
    groups, provider_rejected = _provider_groups(provider_results)
    provider_names = tuple(sorted(groups))
    if not provider_names:
        return FusedCandidates((), tuple(provider_rejected), bound)
    if image_size is None:
        missing_size = list(provider_rejected)
        for provider in provider_names:
            for index, item in enumerate(groups[provider]):
                missing_size.append(RejectedCandidate(_candidate_id(item, index), "missing_image_size", provider))
        return FusedCandidates((), tuple(missing_size), bound)
    default_quota = int(per_source_quota) if per_source_quota is not None else max(1, math.ceil(bound / len(provider_names)))
    if default_quota <= 0:
        raise ValueError("per_source_quota must be positive")
    quotas = {provider: int((source_quotas or {}).get(provider, default_quota)) for provider in provider_names}
    if any(value <= 0 for value in quotas.values()):
        raise ValueError("source quotas must be positive")
    image = image_size
    min_area = float(_param(params, "min_area_ratio", 0.015))
    max_area = float(_param(params, "max_area_ratio", 0.995))
    min_edge = float(_param(params, "min_edge_ratio", 0.03))
    max_edge = float(_param(params, "max_edge_ratio", 1.5))
    queues = {provider: sorted(list(groups[provider]), key=_sort_key) for provider in provider_names}
    positions = {provider: 0 for provider in provider_names}
    accepted_by_provider = {provider: 0 for provider in provider_names}
    fused: list[dict[str, Any]] = []
    rejected: list[RejectedCandidate] = list(provider_rejected)
    seen_ids: set[str] = set()
    flat_index = 0
    redistributed = False
    while len(fused) < bound and any(positions[p] < len(queues[p]) for p in provider_names):
        progressed = False
        for provider in provider_names:
            if len(fused) >= bound or positions[provider] >= len(queues[provider]):
                continue
            # A provider waits at its fair quota while other providers still
            # have work.  Once at least one source is exhausted, redistribute
            # its unused share so the fused pool can still reach 40.
            if not redistributed and accepted_by_provider[provider] >= quotas[provider]:
                continue
            item = queues[provider][positions[provider]]
            positions[provider] += 1
            progressed = True
            cid = _candidate_id(item, flat_index); flat_index += 1
            canonical, reason = _validate_candidate(item, provider, image,
                                                     min_area_ratio=min_area,
                                                     max_area_ratio=max_area,
                                                     min_edge_ratio=min_edge,
                                                     max_edge_ratio=max_edge)
            if reason:
                rejected.append(RejectedCandidate(cid, reason, provider)); continue
            if cid in seen_ids:
                rejected.append(RejectedCandidate(cid, "duplicate_id", provider)); continue
            seen_ids.add(cid)
            match = None
            if image is not None:
                for existing in fused:
                    try:
                        if normalized_corner_distance(canonical["corners"], existing["corners"], image) <= distance:
                            match = existing; break
                    except (GeometryError, TypeError, ValueError):
                        continue
            if match is not None:
                merged_providers = tuple(sorted(set(match.get("providers", ())) | {provider}))
                merged_sources = tuple(sorted(set(match.get("sources", ())) | set(canonical["sources"])))
                merged_provenance = tuple(sorted(tuple(match.get("provenance", ())) + tuple(canonical["provenance"]), key=lambda p: (str(p.get("candidate_id", "")), str(p.get("provider", "")))))
                match["providers"] = merged_providers
                match["sources"] = merged_sources
                match["provenance"] = merged_provenance
                match["score"] = max(float(match.get("score", 0.0) or 0.0), float(canonical.get("score", 0.0) or 0.0))
                rejected.append(RejectedCandidate(cid, "duplicate_geometry", provider))
                continue
            fused.append(canonical)
            accepted_by_provider[provider] += 1
        if not progressed:
            exhausted = any(positions[p] >= len(queues[p]) for p in provider_names)
            if not redistributed and exhausted:
                redistributed = True
                continue
            break
    # Remaining candidates were not considered because the bound/quota was hit;
    # record them explicitly for an auditable, deterministic truncation reason.
    for provider in provider_names:
        while positions[provider] < len(queues[provider]):
            item = queues[provider][positions[provider]]; positions[provider] += 1
            rejected.append(RejectedCandidate(_candidate_id(item, flat_index),
                                               "fused_limit" if len(fused) >= bound else "source_quota", provider))
            flat_index += 1
    return FusedCandidates(tuple(fused), tuple(rejected), bound)


def _score(item: Mapping[str, Any]) -> float:
    for key in ("pre_score", "pre_score_total", "cheap_score", "score"):
        value = item.get(key)
        if value is not None:
            try:
                value = float(value)
                if math.isfinite(value): return value
            except (TypeError, ValueError):
                pass
    for key in ("audit", "pre_truncation_risk_evidence", "stage_scores"):
        nested = item.get(key)
        if isinstance(nested, Mapping):
            for name in ("pre_score", "pre_score_total", "cheap_score", "score"):
                try:
                    value = float(nested[name])
                    if math.isfinite(value): return value
                except (KeyError, TypeError, ValueError):
                    pass
    return 0.0


def _audit_reason(item: Mapping[str, Any]) -> str | None:
    """Return a precondition failure for candidates not audited by Task 9."""
    marker = False
    for key in ("outer_risk", "outer_risk_decision", "risk_decision", "risk",
                "pre_truncation_risk_decisions", "pre_truncation_risk_evidence"):
        if key in item and item[key] not in (None, (), {}, ""):
            marker = True
            break
    nested = item.get("audit")
    if isinstance(nested, Mapping):
        marker = marker or any(key in nested and nested[key] not in (None, (), {}, "")
                               for key in ("outer_risk", "outer_risk_decision", "risk_decision",
                                            "risk", "decision", "evidence", "outer_frame_risk"))
    elif nested is not None:
        return "malformed_audit"
    if not marker:
        return "missing_risk_audit"
    # A cheap finite score is mandatory before Top-K.  ``score`` is accepted
    # for providers that use that public spelling; nested stage scores are also
    # emitted by the scoring task.
    score_keys = ("pre_score", "pre_score_total", "cheap_score", "score")
    finite_score = False
    for key in score_keys:
        if key in item:
            try:
                finite_score = math.isfinite(float(item[key]))
            except (TypeError, ValueError):
                finite_score = False
            if finite_score:
                break
    if not finite_score:
        nested_scores = item.get("stage_scores")
        if isinstance(nested_scores, Mapping):
            for key in ("pre_score", "pre_score_total", "cheap_score", "score"):
                if key in nested_scores:
                    try:
                        finite_score = math.isfinite(float(nested_scores[key]))
                    except (TypeError, ValueError):
                        finite_score = False
                    if finite_score:
                        break
    return None if finite_score else "missing_or_nonfinite_pre_score"


def _audit_present(item: Mapping[str, Any]) -> bool:
    return _audit_reason(item) is None


def _source_key(item: Mapping[str, Any]) -> str:
    values = item.get("sources") or item.get("providers") or item.get("source") or item.get("provider") or "unknown"
    if isinstance(values, str): return values
    try: return sorted(str(value) for value in values)[0]
    except (TypeError, IndexError): return "unknown"


def _near(a: Mapping[str, Any], b: Mapping[str, Any], image: Sequence[float], threshold: float) -> float:
    try:
        return normalized_corner_distance(a["corners"], b["corners"], image)
    except (KeyError, GeometryError, TypeError, ValueError):
        return math.inf


def select_topk(audited_candidates: Iterable[Mapping[str, Any]], image_size: Sequence[float] | None = None,
                top_k: int = 5, *, k: int | None = None,
                dedup_distance: float = 0.015) -> TopKSelection:
    """Select audited candidates with source and geometry diversity.

    This function deliberately refuses raw fused candidates that have no Task 9
    risk/audit marker.  Every candidate omitted after auditing is represented in
    ``truncation_trace`` with a deterministic reason.
    """
    if k is not None: top_k = k
    top_k = int(top_k)
    if top_k <= 0: raise ValueError("top_k must be positive")
    try:
        dedup_distance = float(dedup_distance)
    except (TypeError, ValueError):
        raise ValueError("dedup_distance must be finite and non-negative") from None
    if not math.isfinite(dedup_distance) or dedup_distance < 0:
        raise ValueError("dedup_distance must be finite and non-negative")
    items = []
    for item in audited_candidates:
        if not isinstance(item, Mapping) and hasattr(item, "to_dict"):
            item = item.to_dict()
        if not isinstance(item, Mapping):
            raise ValueError("select_topk requires audited candidates")
        items.append(dict(item))
    precondition_trace: list[dict[str, Any]] = []
    valid_items: list[dict[str, Any]] = []
    for item in items:
        raw_cid = item.get("candidate_id", item.get("id"))
        cid = str(raw_cid) if isinstance(raw_cid, str) else "<invalid>"
        if not isinstance(raw_cid, str) or not raw_cid.strip():
            precondition_trace.append({"stage": "topk_precondition", "candidate_id": cid, "reason": "invalid_candidate_id"})
            continue
        reason = _audit_reason(item)
        if reason is not None:
            precondition_trace.append({"stage": "topk_precondition", "candidate_id": cid, "reason": reason})
        else:
            valid_items.append(item)
    if not valid_items:
        raise ValueError("select_topk requires audited candidates with finite pre-score")
    unique_items: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for item in valid_items:
        cid = str(item.get("candidate_id", item.get("id", "")))
        if cid in seen_candidate_ids:
            precondition_trace.append({"stage": "topk_precondition", "candidate_id": cid,
                                       "reason": "duplicate_candidate_id"})
            continue
        seen_candidate_ids.add(cid)
        unique_items.append(item)
    items = unique_items
    precondition_trace.sort(key=lambda event: str(event.get("candidate_id", "")))
    items.sort(key=lambda item: (-_score(item), str(item.get("candidate_id", item.get("id", "")))))
    if image_size is None:
        image_size = (1000.0, 1000.0)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    # Top-1 is always the globally best audited score.  Diversity is applied
    # only to alternate slots; source iteration must never displace the winner.
    if items:
        first = items[0]
        selected.append(first)
        selected_ids.add(str(first.get("candidate_id", first.get("id", ""))))
    # First cover distinct sources, skipping candidates that are merely geometric
    # near-duplicates of an already covered source.
    for source in sorted({_source_key(item) for item in items}):
        if len(selected) >= top_k: break
        for item in items:
            cid = str(item.get("candidate_id", item.get("id", "")))
            if cid in selected_ids or _source_key(item) != source: continue
            if any(_near(item, prior, image_size, dedup_distance) <= dedup_distance for prior in selected): continue
            selected.append(item); selected_ids.add(cid); break
    # Fill unique geometry clusters first, then use remaining slots for duplicates
    # only when the candidate pool has no additional distinct geometry.
    for allow_near in (False, True):
        if len(selected) >= top_k: break
        for item in items:
            if len(selected) >= top_k: break
            cid = str(item.get("candidate_id", item.get("id", "")))
            if cid in selected_ids: continue
            near = any(_near(item, prior, image_size, dedup_distance) <= dedup_distance for prior in selected)
            if near and not allow_near: continue
            selected.append(item); selected_ids.add(cid)
    trace: list[dict[str, Any]] = list(precondition_trace)
    for rank, item in enumerate(items, 1):
        cid = str(item.get("candidate_id", item.get("id", "")))
        if cid in selected_ids: continue
        nearest = min((_near(item, prior, image_size, dedup_distance) for prior in selected), default=math.inf)
        trace.append({"stage": "topk", "candidate_id": cid,
                      "reason": "near_identical" if nearest <= dedup_distance else "topk_limit",
                      "rank": rank, "nearest_distance": nearest})
    return TopKSelection(tuple(selected), tuple(trace))


reduce_candidates = fuse_candidates
fuse = fuse_candidates

__all__ = ["RejectedCandidate", "FusedCandidates", "TopKSelection", "fuse_candidates", "reduce_candidates", "fuse", "select_topk"]
