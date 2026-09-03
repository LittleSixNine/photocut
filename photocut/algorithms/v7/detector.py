"""Isolated v7 detector orchestration.

The detector is intentionally a small coordinator: input normalization and one
feature context are shared by all providers, while every provider is isolated
behind a structured :class:`ProviderResult`.  The legacy detector is only
loaded when a state-table fallback is actually needed.
"""
from __future__ import annotations

import hashlib
import inspect
import math
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Callable

import numpy as np

from photocut.config import V7_ALGORITHM_VERSION

from .features import FeatureCancelled, ImageFeatureContext
from .input import InputCancelled, InputDecodeError, LoadedImage, decode_bytes, normalize_array
from .parameters import V7Parameters
from .providers import (
    BackgroundDifferenceProvider, ContourProvider, LineProvider,
    WhiteBorderProvider,
)
from .reducer import fuse_candidates
from .refinement import refine_topk
from .scoring import score_candidates, score_scanner_boundaries
from .types import (
    CandidateAudit, DetectionIdentity, DetectionResult, DetectionStatus,
    ProviderResult, ProviderStatus,
)


ALGORITHM_VERSION = V7_ALGORITHM_VERSION


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    for attr in ("is_cancelled", "cancelled", "is_set"):
        value = getattr(token, attr, None)
        if value is not None:
            try:
                return bool(value() if callable(value) else value)
            except (TypeError, ValueError):
                return False
    return bool(token) if isinstance(token, bool) else False


def _cancel_reason(token: Any, default: str) -> str:
    for attr in ("cancellation_reason", "cancel_reason", "reason"):
        value = getattr(token, attr, None) if token is not None else None
        if value not in (None, ""):
            return str(value)
    return default


def _hash_input(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return hashlib.sha256(bytes(value)).hexdigest()
    if isinstance(value, LoadedImage):
        return value.source_sha256
    if hasattr(value, "normalized_bgr"):
        image = np.asarray(value.normalized_bgr)
    else:
        try:
            image = np.asarray(value)
        except Exception:
            image = None
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        return hashlib.sha256(image.tobytes()).hexdigest()
    return hashlib.sha256(repr(value).encode("utf-8", "replace")).hexdigest()


def _identity(value: Any, params: V7Parameters, request_id: Any, image_id: Any, mode: str, orientation: str) -> DetectionIdentity:
    image_hash = _hash_input(value)
    return DetectionIdentity(
        request_id=str(request_id or image_hash[:16]), image_id=str(image_id or image_hash),
        orientation_transform=orientation, algorithm_version=ALGORITHM_VERSION,
        parameter_sha256=params.sha256(), mode=mode,
    )


def _timings(started: float, **extra: float) -> dict[str, float]:
    result = {key: max(0.0, float(value)) for key, value in extra.items()}
    result.setdefault("total", max(0.0, (time.perf_counter() - started) * 1000.0))
    return result


def _result(identity: DetectionIdentity, status: DetectionStatus, *, started: float,
            corners: Any = None, alternate_corners: Any = None,
            overall_confidence: Any = None, edge_confidences: Any = (),
            corner_confidences: Any = (), risks: Any = (), top1_sources: Any = (),
            alternate_sources: Any = (), audits: Any = (), timings: Mapping[str, Any] | None = None,
            debug: Mapping[str, Any] | None = None, error: str | None = None) -> DetectionResult:
    values = dict(timings or {})
    values.setdefault("total", (time.perf_counter() - started) * 1000.0)
    return DetectionResult(identity=identity, status=status, corners=corners,
                           alternate_corners=alternate_corners,
                           overall_confidence=overall_confidence,
                           edge_confidences=tuple(edge_confidences or ()),
                           corner_confidences=tuple(corner_confidences or ()),
                           risks=tuple(dict.fromkeys(str(item) for item in (risks or ()))),
                           top1_sources=tuple(str(item) for item in (top1_sources or ())),
                           alternate_sources=tuple(str(item) for item in (alternate_sources or ())),
                           candidate_audit=tuple(audits or ()), timings_ms=values,
                           debug=dict(debug or {}), error=error)


def detect_corners_v410(image: Any, **kwargs: Any) -> Any:
    # Keep the legacy import lazy.  Importing ``v7`` must not import OpenCV or
    # mutate the v5.2 module surface.
    from photocut.algorithms.v5_2.detector import detect_corners_v410
    return detect_corners_v410(image, return_details=True)


def _default_fallback(image: Any, **kwargs: Any) -> Any:
    # The injected/module-level seam may expose either the v5.2 detailed
    # signature or a minimal test double.  Inspect it once rather than probing
    # with a failed call (the fallback detector must run exactly once).
    try:
        signature = inspect.signature(detect_corners_v410)
        accepts_details = "return_details" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_details = True
    return detect_corners_v410(image, return_details=True) if accepts_details else detect_corners_v410(image)


def _legacy_fallback_disabled(image: Any, **kwargs: Any) -> Any:
    raise RuntimeError("legacy_fallback_disabled")


def _fallback(identity: DetectionIdentity, image: Any, adapter: Callable[..., Any], *, started: float,
              reason: str, provider_results: Sequence[ProviderResult] = ()) -> DetectionResult:
    try:
        # The adapter is invoked exactly once.  Do not retry with a different
        # signature: retrying can run the expensive legacy detector twice.
        # Determine support without probing by invocation; a failed probe would
        # violate the exactly-once fallback contract.  Callable objects with no
        # inspectable signature receive the documented keyword form.
        try:
            signature = inspect.signature(adapter)
            accepts_details = "return_details" in signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_details = True
        output = adapter(image, return_details=True) if accepts_details else adapter(image)
        if isinstance(output, Mapping):
            corners = output.get("corners")
            confidences = output.get("confidences", output.get("corner_confidences", ()))
            details = output.get("debug", output.get("details", {}))
        else:
            if isinstance(output, (tuple, list)) and len(output) == 4 and all(
                isinstance(point, (tuple, list)) and len(point) == 2 for point in output
            ):
                corners, confidences, details = output, (), {}
            else:
                parts = tuple(output) if isinstance(output, (tuple, list)) else (output,)
                corners = parts[0] if parts else None
                confidences = parts[1] if len(parts) > 1 else ()
                details = parts[2] if len(parts) > 2 else {}
        if corners is None:
            raise ValueError("fallback adapter returned no corners")
        source = tuple(str(key) for key in (details.get("sources", ()) if isinstance(details, Mapping) else ()))
        return _result(identity, DetectionStatus.V52_FALLBACK, started=started, corners=corners,
                       edge_confidences=confidences, corner_confidences=confidences,
                       risks=(reason,), top1_sources=source or ("v5.2",),
                       debug={"fallback_reason": reason, "legacy_details": details,
                              "provider_results": tuple(item.to_dict() for item in provider_results)})
    except Exception as exc:
        return _result(identity, DetectionStatus.ERROR, started=started,
                       risks=(reason, "fallback_error"), error=f"{type(exc).__name__}: {exc}",
                       debug={"fallback_reason": reason})


def _normalize(value: Any, params: V7Parameters, token: Any) -> tuple[np.ndarray, str, Any]:
    if isinstance(value, LoadedImage):
        return value.normalized_bgr, value.orientation_transform, value
    if hasattr(value, "normalized_bgr"):
        image = np.asarray(value.normalized_bgr)
        return image, str(getattr(value, "orientation_transform", "identity")), value
    if isinstance(value, (bytes, bytearray, memoryview)):
        loaded = decode_bytes(value, max_pixels=params.max_input_pixels, cancellation_token=token)
        return loaded.normalized_bgr, loaded.orientation_transform, loaded
    image = normalize_array(np.asarray(value))
    return image, "identity", None


def _provider_call(provider: Any, context: ImageFeatureContext, params: V7Parameters, token: Any, deadline: float) -> ProviderResult:
    name = str(getattr(provider, "name", provider.__class__.__name__.lower()))
    started = time.perf_counter()
    try:
        method = getattr(provider, "provide", None) or getattr(provider, "generate", None) or getattr(provider, "run", None)
        if method is None and callable(provider):
            method = provider
        if method is None:
            raise TypeError("provider has no provide method")
        output = method(context, params, cancellation_token=token, deadline=deadline)
        if isinstance(output, ProviderResult):
            return output
        if isinstance(output, Mapping):
            return ProviderResult(name, tuple(output.get("candidates", ())),
                                  status=output.get("status", ProviderStatus.SUCCESS),
                                  elapsed_ms=float(output.get("elapsed_ms", (time.perf_counter() - started) * 1000.0)),
                                  diagnostics=output.get("diagnostics", {}),
                                  timeout_code=output.get("timeout_code"), error_code=output.get("error_code"))
        return ProviderResult(name, tuple(output or ()), status=ProviderStatus.SUCCESS,
                              elapsed_ms=(time.perf_counter() - started) * 1000.0)
    except FeatureCancelled:
        return ProviderResult(name, status=ProviderStatus.CANCELLED,
                              elapsed_ms=(time.perf_counter() - started) * 1000.0, error_code="cancelled")
    except TimeoutError:
        return ProviderResult(name, status=ProviderStatus.TIMEOUT,
                              elapsed_ms=(time.perf_counter() - started) * 1000.0, timeout_code="provider_deadline")
    except Exception as exc:
        return ProviderResult(name, status=ProviderStatus.ERROR,
                              elapsed_ms=(time.perf_counter() - started) * 1000.0,
                              error_code=type(exc).__name__, diagnostics={"message": str(exc)})


def _border_completion_results(results: Sequence[ProviderResult], width: int, height: int) -> tuple[ProviderResult, ...]:
    """Add bounded candidates when a near-full-width frame is cut by a weak mask.

    Several real scans have a uniform white strip at one image edge. The
    background mask can then stop early even though the photo continues to the
    image boundary. Completing only candidates that already span nearly the
    full width keeps this conservative and leaves ordinary interior rectangles
    untouched.
    """
    if width < 2 or height < 2:
        return tuple(results)
    output: list[ProviderResult] = []
    for result in results:
        if result.provider != "background":
            output.append(result)
            continue
        candidates = list(result.candidates)
        additions: list[Mapping[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            corners = candidate.get("corners", candidate.get("original_legal_corners"))
            if not isinstance(corners, (tuple, list)) or len(corners) != 4:
                continue
            try:
                q = tuple((float(point[0]), float(point[1])) for point in corners)
                xmin, xmax = min(p[0] for p in q), max(p[0] for p in q)
                ymin, ymax = min(p[1] for p in q), max(p[1] for p in q)
            except (TypeError, ValueError, IndexError):
                continue
            if (xmax - xmin + 1.0) / width < 0.90:
                continue
            evidence = candidate.get("evidence", {})
            try:
                base_area = float(evidence.get("area_ratio"))
            except (AttributeError, TypeError, ValueError):
                continue
            if not 0.25 <= base_area <= 0.55:
                continue
            top_gap = ymin / height
            bottom_gap = (height - 1.0 - ymax) / height
            if (ymax - ymin + 1.0) / height > 0.90:
                continue
            variants: list[tuple[str, tuple[tuple[float, float], ...]]] = []
            if top_gap >= 0.08:
                variants.append(("top", ((xmin, 0.0), (xmax, 0.0), q[2], q[3])))
            if bottom_gap >= 0.08:
                variants.append(("bottom", (q[0], q[1], (xmax, height - 1.0), (xmin, height - 1.0))))
            if top_gap >= 0.08 and bottom_gap >= 0.08:
                variants.append(("top_bottom", ((xmin, 0.0), (xmax, 0.0), (xmax, height - 1.0), (xmin, height - 1.0))))
            base_id = str(candidate.get("candidate_id", candidate.get("id", "")))
            for variant, completed in variants:
                if not base_id:
                    continue
                cid = hashlib.sha256(f"{base_id}:border_completion:{variant}".encode()).hexdigest()[:20]
                evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
                evidence.update({"derived_variant": f"border_completion_{variant}", "base_candidate_id": base_id})
                completed_candidate = dict(candidate)
                completed_candidate.update({"id": cid, "candidate_id": cid, "corners": completed,
                                            "source": "background:frame_completion",
                                            "sources": ("background:frame_completion",), "evidence": evidence})
                additions.append(completed_candidate)
        if additions:
            candidates.extend(additions[:24])
            output.append(ProviderResult(result.provider, tuple(candidates), result.status,
                                         result.elapsed_ms, result.work_consumed, result.work_limit,
                                         result.timeout_code, result.error_code, result.diagnostics))
        else:
            output.append(result)
    return tuple(output)


def _affirmative_no_primary(results: Sequence[ProviderResult], fused: Sequence[Mapping[str, Any]], scored: Any = None, params: V7Parameters | None = None) -> tuple[bool, str | None]:
    for result in results:
        diagnostics = result.diagnostics
        if not isinstance(diagnostics, Mapping):
            continue
        if diagnostics.get("affirmative_no_photo") or diagnostics.get("no_photo"):
            return True, "no_photo_evidence"
        if diagnostics.get("affirmative_multiple_primary") or diagnostics.get("multiple_primary"):
            return True, "multiple_primary_ambiguity"
        marker = str(diagnostics.get("primary_ambiguity", "")).lower()
        if marker in {"multiple_primary", "multiple_photos", "multiple_separate_photos", "no_photo"}:
            return True, "no_photo_evidence" if marker == "no_photo" else "multiple_primary_ambiguity"
    if not scored:
        return False, None
    ranked = sorted(tuple(getattr(scored, "candidates", ()) or getattr(scored, "selected", ())),
                    key=lambda item: (-float(item.get("score", item.get("pre_score", 0.0)) or 0.0),
                                      str(item.get("candidate_id", item.get("id", "")))))
    if len(ranked) < 2:
        return False, None
    first, second = ranked[0], ranked[1]
    try:
        score_gap = abs(float(first.get("score", first.get("pre_score", 0.0))) - float(second.get("score", second.get("pre_score", 0.0))))
        a, b = first["corners"], second["corners"]
        aw = max(abs(float(p[0])) for p in a) + 1.0; bw = max(abs(float(p[0])) for p in b) + 1.0
        center_a = (sum(float(p[0]) for p in a) / 4.0, sum(float(p[1]) for p in a) / 4.0)
        center_b = (sum(float(p[0]) for p in b) / 4.0, sum(float(p[1]) for p in b) / 4.0)
        separation = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]) / max(aw, bw)
        audits = {audit.candidate_id: audit for audit in scored.audits}
        nested = audits.get(str(first.get("candidate_id", "")))
        nested_second = audits.get(str(second.get("candidate_id", "")))
        nested_flag = bool((nested and "suspected_outer_frame" in nested.pre_truncation_risk_decisions) or
                           (nested_second and "suspected_outer_frame" in nested_second.pre_truncation_risk_decisions))
        threshold = float((params or V7Parameters()).risk_thresholds.get("ambiguity", 0.08))
        if not nested_flag and score_gap <= threshold and separation > 0.18 and min(float(first.get("score", 0.0)), float(second.get("score", 0.0))) >= .62:
            return True, "multiple_primary_ambiguity"
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        pass
    return False, None


def _blank_photo_evidence(image: np.ndarray) -> bool:
    """Conservative affirmative evidence for a blank/no-photo scan."""
    try:
        spread = float(np.std(image)); mean = float(np.mean(image))
        return spread < 1.0 and (mean < 2.0 or mean > 253.0)
    except (TypeError, ValueError):
        return False


def _supplemental_audits(
    provider_results: Sequence[ProviderResult], *, context: ImageFeatureContext,
    image_size: Sequence[float], start_rank: int = 6,
) -> tuple[CandidateAudit, ...]:
    """Project white-border proposals into the cascade audit trail only."""
    candidates = []
    for result in provider_results:
        if result.provider != "white_border" or result.status is not ProviderStatus.SUCCESS:
            continue
        candidates.extend(item for item in result.candidates if isinstance(item, Mapping))
    candidates.sort(key=lambda item: (
        -float(item.get("score", 0.0) or 0.0),
        str(item.get("candidate_id", item.get("id", ""))),
    ))
    candidates = candidates[:3]
    boundary_by_id = score_scanner_boundaries(
        candidates, context, image_size=image_size,
    ) if candidates else {}
    audits = []
    for offset, item in enumerate(candidates):
        candidate_id = str(item.get("candidate_id", item.get("id", "")))
        corners = item.get("corners", item.get("original_legal_corners"))
        sources = item.get("sources", (item.get("source"),))
        if isinstance(sources, str):
            sources = (sources,)
        evidence = item.get("evidence")
        components = dict(boundary_by_id.get(candidate_id, {}))
        audits.append(CandidateAudit(
            candidate_id=candidate_id,
            sources=tuple(str(value) for value in sources if value),
            original_legal_corners=corners,
            pre_topk_corners=corners,
            pre_truncation_risk_decisions=("none",),
            pre_truncation_risk_evidence=(
                dict(evidence) if isinstance(evidence, Mapping) else {}
            ),
            stage_scores={
                "full_score": float(item.get("score", 0.0) or 0.0),
                "provider_stability": float(item.get("score", 0.0) or 0.0),
                "components": components,
            },
            stage_ranks={"selected": start_rank + offset},
            truncation_stage="selected",
            truncation_reason="scanner_white_supplement",
        ))
    return tuple(audits)


def _add_boundary_scores(
    audits: Sequence[CandidateAudit], candidates: Sequence[Mapping[str, Any]],
    *, context: ImageFeatureContext, image_size: Sequence[float],
) -> tuple[CandidateAudit, ...]:
    scores = score_scanner_boundaries(
        candidates, context, image_size=image_size,
    )
    output = []
    for audit in audits:
        values = scores.get(audit.candidate_id)
        if not values:
            output.append(audit)
            continue
        data = audit.to_dict()
        stage_scores = dict(data.get("stage_scores", {}))
        components = dict(stage_scores.get("components", {}))
        components.update(values)
        stage_scores["components"] = components
        data["stage_scores"] = stage_scores
        output.append(CandidateAudit.from_dict(data))
    return tuple(output)


def detect_corners_v7(image: Any, *, params: V7Parameters | None = None,
                      providers: Sequence[Any] | None = None,
                      cancellation_token: Any = None, cancel_token: Any = None,
                      fallback_adapter: Callable[..., Any] | None = None,
                      allow_legacy_fallback: bool = True,
                      request_id: str | None = None, image_id: str | None = None,
                      mode: str | None = None) -> DetectionResult:
    """Run the bounded v7 pipeline and return an immutable result."""
    started = time.perf_counter()
    token = cancellation_token if cancellation_token is not None else cancel_token
    params = params or V7Parameters(mode=mode or "safe")
    if mode is not None and mode != params.mode:
        params = params.replace(mode=mode)
    fallback = fallback_adapter or (_default_fallback if allow_legacy_fallback else _legacy_fallback_disabled)
    orientation = "identity"
    try:
        orientation = getattr(image, "orientation_transform", "identity")
        identity = _identity(image, params, request_id, image_id, params.mode, str(orientation))
    except Exception as exc:
        # Identity construction itself must remain immutable and explicit.
        identity = DetectionIdentity(str(request_id or "invalid"), _hash_input(image), "unknown", ALGORITHM_VERSION, params.sha256(), params.mode)
        return _result(identity, DetectionStatus.ERROR, started=started, error=f"{type(exc).__name__}: {exc}")
    if _cancelled(token):
        reason = _cancel_reason(token, "cancelled_before_start")
        return _result(identity, DetectionStatus.CANCELLED, started=started,
                       risks=("cancelled",), error=reason, debug={"cancellation_reason": reason})
    try:
        normalized, orientation, loaded = _normalize(image, params, token)
        if loaded is not None and orientation != identity.orientation_transform:
            identity = DetectionIdentity(identity.request_id, identity.image_id, str(orientation), identity.algorithm_version, identity.parameter_sha256, identity.mode)
    except (InputCancelled, FeatureCancelled):
        reason = _cancel_reason(token, "input")
        return _result(identity, DetectionStatus.CANCELLED, started=started,
                       risks=("cancelled",), error=reason, debug={"cancellation_reason": reason})
    except (InputDecodeError, TypeError, ValueError) as exc:
        return _fallback(identity, image, fallback, started=started,
                         reason="unsupported_input", provider_results=())
    context = ImageFeatureContext(normalized, cancellation_token=token)
    provider_results: list[ProviderResult] = []
    try:
        if providers is not None:
            active = tuple(providers)
        else:
            active = (BackgroundDifferenceProvider(), ContourProvider(), LineProvider())
            if params.scene_profile == "scanner_white":
                active += (WhiteBorderProvider(),)
        for provider in active:
            if _cancelled(token):
                reason = _cancel_reason(token, "provider")
                return _result(identity, DetectionStatus.CANCELLED, started=started, risks=("cancelled",),
                               error=reason, debug={"cancellation_reason": reason})
            name = str(getattr(provider, "name", provider.__class__.__name__.lower()))
            timeout_ms = int(params.provider_timeout_ms.get(name, 250))
            provider_results.append(_provider_call(provider, context, params, token, time.monotonic() + timeout_ms / 1000.0))
        if _cancelled(token):
            reason = _cancel_reason(token, "provider")
            return _result(identity, DetectionStatus.CANCELLED, started=started, risks=("cancelled",), error=reason,
                           debug={"cancellation_reason": reason})
        provider_cancelled = next((item for item in provider_results if item.status is ProviderStatus.CANCELLED), None)
        if provider_cancelled is not None:
            reason = provider_cancelled.error_code or "provider_cancelled"
            return _result(identity, DetectionStatus.CANCELLED, started=started, risks=("cancelled",), error=reason,
                           debug={"cancellation_reason": reason,
                                  "provider_results": tuple(item.to_dict() for item in provider_results)})
        primary_provider_results = tuple(
            item for item in provider_results if item.provider != "white_border"
        )
        successful = tuple(item for item in primary_provider_results if item.status is ProviderStatus.SUCCESS and item.candidates)
        failures = tuple(item for item in primary_provider_results if item.status in {ProviderStatus.TIMEOUT, ProviderStatus.BUDGET_EXHAUSTED, ProviderStatus.ERROR})
        affirmative, affirmative_reason = _affirmative_no_primary(primary_provider_results, ())
        if (not affirmative and not successful and not failures and
                primary_provider_results and all(item.status is ProviderStatus.NO_CANDIDATE for item in primary_provider_results) and
                _blank_photo_evidence(normalized)):
            affirmative, affirmative_reason = True, "no_photo_evidence"
        if affirmative:
            return _result(identity, DetectionStatus.NO_PRIMARY_PHOTO, started=started,
                           risks=(affirmative_reason or "no_primary_photo",),
                           error=affirmative_reason or "no_primary_photo",
                           debug={"provider_results": tuple(item.to_dict() for item in provider_results)})
        if not successful and failures:
            return _fallback(identity, normalized, fallback, started=started, reason="all_providers_failed", provider_results=provider_results)
        height, width = normalized.shape[:2]
        successful = _border_completion_results(successful, width, height)
        fused = fuse_candidates(successful, image_size=(float(width), float(height)), params=params, fused_limit=min(40, params.fused_budget))
        if not fused.candidates:
            affirmative, why = _affirmative_no_primary(primary_provider_results, fused.candidates)
            if not affirmative and primary_provider_results and _blank_photo_evidence(normalized):
                affirmative, why = True, "no_photo_evidence"
            if affirmative:
                return _result(identity, DetectionStatus.NO_PRIMARY_PHOTO, started=started, risks=(why or "no_primary_photo",),
                               error=why or "no_primary_photo", debug={"provider_results": tuple(item.to_dict() for item in provider_results), "rejected": fused.rejected_reasons})
            return _fallback(identity, normalized, fallback, started=started, reason="zero_legal_candidates", provider_results=provider_results)
        scored = score_candidates(fused.candidates, context, (float(width), float(height)), top_k=5, params=params)
        if not scored.selected:
            return _fallback(identity, normalized, fallback, started=started, reason="zero_legal_candidates", provider_results=provider_results)
        refinements = refine_topk(context, scored.selected, params, cancellation_token=token)
        refined_by_id = {item.candidate_id: item for item in refinements}
        final_items = []
        for item in scored.selected:
            refinement = refined_by_id.get(str(item.get("candidate_id", item.get("id", ""))))
            updated = dict(item)
            if refinement is not None:
                updated["corners"] = refinement.adopted_corners
                updated["refinement"] = refinement
            final_items.append(updated)
        # Full score/rank remains deterministic and mode-independent.  Re-score
        # the refined Top-K only; Top-1 identity/order is never mode-dependent.
        rescored = score_candidates(final_items, context, (float(width), float(height)), top_k=5, params=params)
        if params.scene_profile == "scanner_white":
            primary_audits = list(_add_boundary_scores(
                rescored.audits, rescored.selected, context=context,
                image_size=(float(width), float(height)),
            ))
            audits = primary_audits + list(_supplemental_audits(
                provider_results, context=context,
                image_size=(float(width), float(height)),
            ))
        else:
            primary_audits = list(rescored.audits)
            audits = primary_audits
        risks: list[str] = []
        # A partial provider failure is retained as an explicit safety risk;
        # it never deletes complete candidates from other providers.
        for provider_result in failures:
            if provider_result.status is ProviderStatus.TIMEOUT:
                risks.append("provider_timeout")
            elif provider_result.status is ProviderStatus.BUDGET_EXHAUSTED:
                risks.append("provider_budget_exhausted")
            elif provider_result.status is ProviderStatus.ERROR:
                risks.append("provider_error")
        for audit in primary_audits:
            risks.extend(str(value) for value in audit.pre_truncation_risk_decisions if value != "none")
        for refinement in refinements:
            risks.extend(refinement.risks)
        # Rank from the final (post-refinement/full) score, with stable ID as
        # the sole tie-break.  Never carry the pre-score Top-K order forward.
        final_ranked = sorted(
            tuple(rescored.selected),
            key=lambda item: (
                -float(item.get("score", item.get("pre_score", 0.0)) or 0.0),
                str(item.get("candidate_id", item.get("id", ""))),
            ),
        )
        top1 = final_ranked[0]
        alternate = final_ranked[1] if len(final_ranked) > 1 else None
        ambiguity, why = _affirmative_no_primary(primary_provider_results, fused.candidates, rescored, params)
        if ambiguity:
            return _result(identity, DetectionStatus.NO_PRIMARY_PHOTO, started=started, risks=tuple(risks) + (why or "multiple_primary_ambiguity",),
                           error=why or "multiple_primary_ambiguity", audits=audits,
                           debug={"provider_results": tuple(item.to_dict() for item in provider_results)})
        components = top1.get("component_scores", {})
        confidence = float(np.clip(top1.get("score", top1.get("pre_score", 0.0)), 0.0, 1.0))
        edge = tuple(float(np.clip(value, 0.0, 1.0)) for value in components.get("edge_scores", ()))
        mandatory = tuple(dict.fromkeys(risks))
        threshold = params.safe_threshold if params.mode == "safe" else params.aggressive_threshold
        status = DetectionStatus.V7_RECOMMENDED if confidence >= threshold and not mandatory else DetectionStatus.V7_LOW_CONFIDENCE
        return _result(identity, status, started=started, corners=top1.get("corners"),
                       alternate_corners=alternate.get("corners") if alternate else None,
                       overall_confidence=confidence, edge_confidences=edge,
                       corner_confidences=tuple(edge), risks=mandatory,
                       top1_sources=top1.get("sources", top1.get("providers", ())),
                       alternate_sources=alternate.get("sources", ()) if alternate else (), audits=audits,
                       timings={"features": sum(context.timings_ms.values()), "providers": sum(item.elapsed_ms for item in provider_results)},
                       debug={"provider_results": tuple(item.to_dict() for item in provider_results), "fused_count": len(fused.candidates)})
    except (FeatureCancelled, InputCancelled):
        reason = _cancel_reason(token, "pipeline")
        return _result(identity, DetectionStatus.CANCELLED, started=started, risks=("cancelled",), error=reason,
                       debug={"cancellation_reason": reason})
    except Exception as exc:
        return _result(identity, DetectionStatus.ERROR, started=started, risks=("unrecoverable_error",),
                       error=f"{type(exc).__name__}: {exc}",
                       debug={"provider_results": tuple(item.to_dict() for item in provider_results)})
    finally:
        context.close()


__all__ = ["ALGORITHM_VERSION", "detect_corners_v7", "detect_corners_v410"]
