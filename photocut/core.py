# -*- coding: utf-8 -*-
"""
photocut 核心模块
整合四角检测和透视裁剪
"""

import cv2
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from photocut.algorithms.v5_2.detector import detect_corners_detailed
from photocut.data.dataset_store import atomic_write_json
from photocut.geometry.perspective import perspective_transform
from photocut.config import (
    ALGORITHM_VERSION,
    CORNERS_INFO_FILE,
    DEFAULT_SCENE_PROFILE,
    OUTPUT_DIR,
    PREVIEW_MAX_SIZE,
    V7_ALGORITHM_VERSION,
)
from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS, DetectionParameters


def _project_v7_result(result: Any, img: np.ndarray, *, img_path: str,
                       relative_path: Optional[str] = None,
                       parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Project an immutable v7 result into the legacy mutable entry shape.

    The projection is intentionally one-way: v7 keeps normalized floating point
    corners and its immutable audit, while the GUI/crop code continues to read
    the established ``corners_info.json`` fields.  States without a valid
    polygon are explicitly unsuccessful and therefore cannot be cropped.
    """
    from photocut.algorithms.v7.types import DetectionStatus

    h, w = img.shape[:2]
    preview = create_preview(img)
    preview_h, preview_w = preview.shape[:2]
    status = result.status
    corners = result.corners if status in {
        DetectionStatus.V7_RECOMMENDED,
        DetectionStatus.V7_LOW_CONFIDENCE,
        DetectionStatus.V52_FALLBACK,
    } else None
    normalized = [
        [float(point[0]), float(point[1])] for point in (corners or ())
    ]
    valid = len(normalized) == 4
    preview_corners = [
        [int(round(point[0] * preview_w / w)), int(round(point[1] * preview_h / h))]
        for point in normalized
    ] if valid and w and h else []
    identity = result.identity.to_dict()
    status_value = status.value
    fallback = status is DetectionStatus.V52_FALLBACK
    confidence = result.overall_confidence
    if confidence is None and result.edge_confidences:
        confidence = float(sum(result.edge_confidences) / len(result.edge_confidences))
    debug = result.to_dict()
    return {
        "filename": relative_path or os.path.basename(img_path),
        "algorithm_boundary_corners": normalized,
        "boundary_corners": [point[:] for point in normalized],
        "algorithm_version": identity["algorithm_version"],
        "algorithm_corners": [point[:] for point in normalized],
        "algorithm_preview_corners": [point[:] for point in preview_corners],
        "manual_corners": None,
        "manual_preview_corners": None,
        "corners": [point[:] for point in normalized],
        "preview_corners": [point[:] for point in preview_corners],
        "crop_corners": None,
        "inset": None,
        "original_size": [w, h],
        "preview_size": [preview_w, preview_h],
        "success": bool(valid),
        "manually_adjusted": False,
        "algorithm_generated": True,
        "confirmed": False,
        "error_message": result.error,
        "adjust_count": 0,
        "adjust_timestamp": None,
        # v7-only immutable identity/audit projection fields.
        "detector": "v7",
        "detector_requested": "v7",
        "detector_used": "v7",
        "detection_id": identity["request_id"],
        "detection_identity": identity,
        "detection_parameters": dict(parameters or {
            "mode": identity["mode"],
            "parameter_sha256": identity["parameter_sha256"],
        }),
        "detection_status": status_value,
        "v7_mode": identity["mode"],
        "normalized_orientation": identity["orientation_transform"],
        "overall_confidence": confidence,
        "edge_confidences": list(result.edge_confidences),
        "corner_confidences": list(result.corner_confidences),
        "confidences": list(result.corner_confidences or result.edge_confidences),
        "risks": list(result.risks),
        "candidate_sources": list(result.top1_sources),
        "alternate_corners": [list(point) for point in result.alternate_corners] if result.alternate_corners else None,
        "alternate_sources": list(result.alternate_sources),
        "candidate_audit": debug.get("candidate_audit", []),
        "detection_timings_ms": dict(result.timings_ms),
        "detection_debug": debug,
        "fallback_reason": (result.error or (result.risks[0] if result.risks else None)) if fallback else None,
        "legacy_fallback": fallback,
    }


def detect_and_save_corners_v7(
    img_path: str,
    output_dir: str,
    *,
    img: Optional[np.ndarray] = None,
    v7_mode: str = "safe",
    scene_profile: str = DEFAULT_SCENE_PROFILE,
    relative_path: Optional[str] = None,
    request_id: Optional[str] = None,
    image_id: Optional[str] = None,
    fallback_adapter: Any = None,
    loaded_input: Any = None,
    allow_legacy_fallback: bool = False,
) -> Dict[str, Any]:
    """Run the opt-in v7 detector and project its result for the GUI.

    This adapter is never called by the legacy path.  Imports are lazy so a
    normal v5.2 invocation does not load the v7 package or alter its behavior.
    """
    if loaded_input is not None:
        img = loaded_input.normalized_bgr
    elif img is None and os.path.isfile(img_path):
        # v7 owns byte decoding so EXIF orientation and TIFF metadata are
        # normalized before detection.  Decode to a bounded analysis image so
        # an oversized scan never falls through to the legacy full-image
        # loader.  Tests and callers may still inject an already-decoded array.
        try:
            from photocut.algorithms.v7.input import decode_bytes_for_analysis
            from photocut.algorithms.v7.parameters import V7Parameters
            params = V7Parameters(mode=v7_mode, scene_profile=scene_profile)
            loaded_input = decode_bytes_for_analysis(
                Path(img_path).read_bytes(),
                max_pixels=params.max_input_pixels,
                analysis_max_edge=4096,
            )
            img = loaded_input.normalized_bgr
        except Exception:
            img = None
    if img is None:
        return {
            "filename": relative_path or os.path.basename(img_path),
            "algorithm_version": V7_ALGORITHM_VERSION,
            "detector": "v7",
            "v7_mode": v7_mode,
            "scene_profile": scene_profile,
            "detection_status": "error",
            "success": False,
            "confirmed": False,
            "algorithm_generated": True,
            "error_message": "Failed to load image",
            "algorithm_boundary_corners": [], "boundary_corners": [],
            "algorithm_corners": [], "corners": [], "preview_corners": [],
            "manual_corners": None, "manual_preview_corners": None,
            "detection_debug": [], "confidences": [],
        }
    from photocut.algorithms.v7.detector import detect_corners_v7
    from photocut.algorithms.v7.parameters import V7Parameters

    params = V7Parameters(mode=v7_mode, scene_profile=scene_profile)
    result = detect_corners_v7(
        loaded_input if loaded_input is not None else img,
        params=params,
        request_id=request_id,
        image_id=image_id,
        fallback_adapter=fallback_adapter,
        allow_legacy_fallback=allow_legacy_fallback,
        mode=v7_mode,
    )
    projected = _project_v7_result(
        result, img, img_path=img_path, relative_path=relative_path,
        parameters=params.to_dict(),
    )
    projected["scene_profile"] = scene_profile
    if loaded_input is not None:
        projected.update({
            "source_sha256": loaded_input.source_sha256,
            "source_original_size": list(loaded_input.original_size),
            "normalized_orientation": loaded_input.orientation_transform,
        })
        projected = _map_analysis_entry_to_full(projected, loaded_input)
    return projected


def _v8_fallback_result(core_payload: Dict[str, Any], provider_status: str,
                        reason: str, *, cancelled: bool = False) -> Any:
    """Keep the V7 core untouched and place V8 request data outside it."""
    from photocut.algorithms.v8.cascade import V8CascadeStatus, V8DormantResult

    status = V8CascadeStatus.CANCELLED if cancelled else V8CascadeStatus.V7_FALLBACK
    return V8DormantResult(
        core_payload=core_payload,
        audit_envelope={
            "requested_detector": "v8",
            "v8_provider_status": provider_status,
            "v8_fallback_reason": reason,
        },
        status=status,
    )


def _v8_candidate_corners(candidate: Any) -> Any:
    if not isinstance(candidate, dict):
        try:
            candidate = dict(candidate)
        except (TypeError, ValueError):
            return None
    return (
        candidate.get("adopted_refined_corners")
        or candidate.get("pre_topk_corners")
        or candidate.get("original_legal_corners")
        or candidate.get("corners")
    )


def _v8_core_from_v7(core_payload: Dict[str, Any], loaded: Any, decision: Any,
                      policy: Any) -> Dict[str, Any]:
    """Project one analysis-space V8 decision without mutating the V7 draft."""
    import copy
    from photocut.algorithms.v8.cascade import V8CascadeStatus

    output = copy.deepcopy(core_payload)
    full_corners = loaded.map_analysis_to_full(decision.selected_corners)
    corners = [[float(x), float(y)] for x, y in full_corners]
    for field in (
        "algorithm_boundary_corners", "boundary_corners", "algorithm_corners", "corners",
    ):
        output[field] = [point[:] for point in corners]

    original_size = output.get("original_size") or list(loaded.full_normalized_size)
    preview_size = output.get("preview_size") or original_size
    if (
        isinstance(original_size, (list, tuple)) and len(original_size) == 2
        and isinstance(preview_size, (list, tuple)) and len(preview_size) == 2
        and int(original_size[0]) > 0 and int(original_size[1]) > 0
    ):
        preview = [[
            int(round(point[0] * int(preview_size[0]) / int(original_size[0]))),
            int(round(point[1] * int(preview_size[1]) / int(original_size[1]))),
        ] for point in corners]
    else:
        preview = []
    output["algorithm_preview_corners"] = [point[:] for point in preview]
    output["preview_corners"] = [point[:] for point in preview]
    output["algorithm_version"] = policy.algorithm_version
    output["detector"] = "v8"
    output["detector_requested"] = "v8"
    output["detector_used"] = (
        "v8" if decision.status is V8CascadeStatus.AUTOMATIC else "manual_review"
    )
    output["detection_status"] = (
        "v8_recommended"
        if decision.status is V8CascadeStatus.AUTOMATIC
        else "manual_review"
    )
    output["success"] = True
    output["confirmed"] = False
    output["legacy_fallback"] = False
    output["fallback_reason"] = None
    output["candidate_sources"] = [decision.selected_candidate_id]
    output["v8_policy_sha256"] = policy.policy_sha256
    output["v8_selection_reason"] = decision.reason
    output["v8_cascade_evidence"] = dict(decision.evidence)
    output["detection_parameters"] = policy.parameters.to_dict()
    identity = dict(output.get("detection_identity") or {})
    identity.update({
        "algorithm_version": policy.algorithm_version,
        "parameter_sha256": policy.policy_sha256.removeprefix("sha256:"),
    })
    output["detection_identity"] = identity
    debug = dict(output.get("detection_debug") or {})
    debug["v8_cascade"] = decision.to_dict()
    output["detection_debug"] = debug
    return output


def detect_and_save_corners_v8_dormant(
    img_path: str,
    output_dir: str,
    *,
    policy: Any,
    mask_provider: Any,
    runtime_config: Any,
    runtime_config_sha256: Optional[str] = None,
    model_manifest_sha256: Optional[str] = None,
    boundary_quality_artifact: Any = None,
    loaded_input: Any = None,
    scene_profile: str = "scanner_white",
    relative_path: Optional[str] = None,
    request_id: Optional[str] = None,
    image_id: Optional[str] = None,
    cancellation_token: Any = None,
    deadline: Any = None,
) -> Any:
    """Run the sealed V8 release candidate without exposing any CLI route.

    The current default and GUI never call this function.  It exists so the
    future one-shot validation and, only after promotion, the CLI integration
    execute the exact same bounded algorithm core.
    """
    from photocut.algorithms.v7.features import ImageFeatureContext
    from photocut.algorithms.v7.input import LoadedImage, decode_bytes_for_analysis
    from photocut.algorithms.v7.parameters import V7Parameters
    from photocut.algorithms.v7.types import ProviderResult, ProviderStatus
    from photocut.algorithms.v8.cascade import V8CascadeStatus, V8DormantResult, decide_v8_cascade
    from photocut.algorithms.v8.code_seal import compute_validated_core_sha256
    from photocut.algorithms.v8.edge_hypotheses import aggregate_edge_hypotheses, generate_edge_hypotheses
    from photocut.algorithms.v8.parameters import canonical_sha256
    from photocut.algorithms.v8.policy import V8Policy
    from photocut.algorithms.v8.boundary_quality import (
        BoundaryQualityArtifact,
        evaluate_boundary_quality,
        serialize_ranked_candidate,
    )
    from photocut.algorithms.v8.scanner_selector import rank_scanner_candidates, select_scanner_candidate
    from photocut.algorithms.v8.seed_selection import select_edge_seed_candidates
    from photocut.algorithms.v8.truth_coordinates import map_full_normalized_to_analysis

    if scene_profile != "scanner_white":
        raise ValueError("dormant V8 adapter only supports scanner_white")
    if not isinstance(policy, V8Policy):
        raise TypeError("policy must be a verified V8Policy")
    if policy.parameters.scene_profile != scene_profile:
        raise ValueError("policy scene profile mismatch")

    loaded = loaded_input
    if loaded is not None and not isinstance(loaded, LoadedImage):
        raise TypeError("loaded_input must be a v7 LoadedImage")
    if loaded is None:
        params = V7Parameters(scene_profile=scene_profile)
        try:
            loaded = decode_bytes_for_analysis(
                Path(img_path).read_bytes(),
                max_pixels=params.max_input_pixels,
                analysis_max_edge=4096,
                cancellation_token=cancellation_token,
            )
        except Exception:
            core = detect_and_save_corners_v7(
                img_path,
                output_dir,
                scene_profile=scene_profile,
                relative_path=relative_path,
                request_id=request_id,
                image_id=image_id,
                allow_legacy_fallback=False,
            )
            return _v8_fallback_result(core, "not_run", "input_decode_error")

    core = detect_and_save_corners_v7(
        img_path,
        output_dir,
        scene_profile=scene_profile,
        relative_path=relative_path,
        request_id=request_id,
        image_id=image_id,
        loaded_input=loaded,
        allow_legacy_fallback=False,
    )
    if (
        policy.schema_version == 3
        and core.get("detection_status") == "error"
        and not core.get("corners")
    ):
        # V7 providers use short wall-clock guards.  Under transient local
        # contention they can all expire even though the already-decoded image
        # is valid.  Explicit V8.2 gets one bounded retry before mask work;
        # default auto and successful V7 results remain single-pass.
        core = detect_and_save_corners_v7(
            img_path,
            output_dir,
            scene_profile=scene_profile,
            relative_path=relative_path,
            request_id=request_id,
            image_id=image_id,
            loaded_input=loaded,
            allow_legacy_fallback=False,
        )
    project_root = Path(__file__).resolve().parent
    if compute_validated_core_sha256(project_root) != policy.validated_core_sha256:
        return _v8_fallback_result(core, "not_run", "validated_core_mismatch")
    if policy.schema_version == 3:
        if (
            not isinstance(boundary_quality_artifact, BoundaryQualityArtifact)
            or boundary_quality_artifact.source_sha256
            != policy.boundary_quality_artifact_sha256
        ):
            return _v8_fallback_result(
                core, "not_run", "boundary_quality_artifact_mismatch"
            )
    expected_runtime_config = {
        "schema_version": 1,
        "provider": "CPUExecutionProvider",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
    }
    runtime_identity = runtime_config_sha256 or (
        canonical_sha256(runtime_config) if isinstance(runtime_config, dict) else None
    )
    if runtime_config != expected_runtime_config or runtime_identity != policy.runtime_config_sha256:
        return _v8_fallback_result(core, "not_run", "runtime_config_mismatch")
    if mask_provider is None:
        return _v8_fallback_result(core, "not_run", "mask_provider_unavailable")
    manifest = getattr(mask_provider, "manifest", None)
    manifest_identity = model_manifest_sha256 or getattr(manifest, "manifest_sha256", None)
    if (
        getattr(manifest, "model_id", None) != policy.model_id
        or getattr(manifest, "model_sha256", None) != policy.model_sha256
        or manifest_identity != policy.model_manifest_sha256
        or float(getattr(mask_provider, "threshold", -1.0)) != float(policy.parameters.mask_threshold)
    ):
        return _v8_fallback_result(core, "not_run", "model_identity_mismatch")

    v7_params = V7Parameters(scene_profile=scene_profile)
    context = ImageFeatureContext.from_loaded(loaded, cancellation_token)
    mask_probability = None
    try:
        if policy.schema_version == 3:
            provide_with_probability = getattr(
                mask_provider, "provide_with_probability", None
            )
            if not callable(provide_with_probability):
                return _v8_fallback_result(
                    core, "not_run", "mask_probability_unavailable"
                )
            mask_result, mask_probability = provide_with_probability(
                context,
                v7_params,
                cancellation_token=cancellation_token,
                deadline=deadline,
            )
        else:
            mask_result = mask_provider.provide(
                context,
                v7_params,
                cancellation_token=cancellation_token,
                deadline=deadline,
            )
    except Exception:
        return _v8_fallback_result(core, "error", "mask_provider_error")
    if not isinstance(mask_result, ProviderResult):
        return _v8_fallback_result(core, "error", "invalid_mask_provider_result")
    if mask_result.status in {
        ProviderStatus.ERROR,
        ProviderStatus.TIMEOUT,
        ProviderStatus.BUDGET_EXHAUSTED,
        ProviderStatus.CANCELLED,
    }:
        reason = f"mask_provider_{mask_result.status.value}"
        return _v8_fallback_result(
            core,
            mask_result.status.value,
            reason,
            cancelled=mask_result.status is ProviderStatus.CANCELLED,
        )
    if (
        policy.schema_version == 3
        and mask_result.status is ProviderStatus.NO_CANDIDATE
        and dict(mask_result.diagnostics).get("reason")
        == "mask_candidate_refinement_invalid"
    ):
        return _v8_fallback_result(
            core,
            mask_result.status.value,
            "mask_candidate_refinement_invalid",
        )

    audit = core.get("candidate_audit")
    audit = audit if isinstance(audit, list) else []
    try:
        seeds = select_edge_seed_candidates(audit, policy.parameters.edge_seed_config)
        edge_observations = []
        for seed in seeds:
            full_corners = _v8_candidate_corners(seed)
            if full_corners is None:
                continue
            analysis_corners = map_full_normalized_to_analysis(loaded, full_corners)
            generated = generate_edge_hypotheses(
                loaded.normalized_bgr,
                analysis_corners,
                policy.parameters.edge_config,
            )
            seed_id = str(seed["candidate_id"])
            raw_sources = seed.get("sources", ())
            seed_sources = (
                (raw_sources,)
                if isinstance(raw_sources, str)
                else tuple(str(value) for value in raw_sources)
            )
            for candidate in generated.candidates:
                edge_observations.append((seed_id, seed_sources, candidate))
        edge_candidates = list(aggregate_edge_hypotheses(
            edge_observations,
            budget=policy.parameters.edge_budget,
        ))

        v7_corners = core.get("corners")
        v7_candidate = None
        if isinstance(v7_corners, (list, tuple)) and len(v7_corners) == 4:
            v7_candidate = {
                "candidate_id": "v7:current",
                "corners": map_full_normalized_to_analysis(loaded, v7_corners),
                "prior_score": float(core.get("overall_confidence") or 0.5),
            }
        mask_candidate = mask_result.candidates[0] if mask_result.candidates else None
        selection = select_scanner_candidate(
            loaded.normalized_bgr,
            edge_candidates,
            v7_candidate=v7_candidate,
            mask_candidate=mask_candidate,
            config=policy.parameters.exterior_config,
            selector_config=policy.parameters.selector_config,
        )
        decision = decide_v8_cascade(
            selection,
            provider_status=mask_result.status,
            policy=policy,
            image_size=tuple(loaded.normalized_size),
            v7_candidate=v7_candidate,
            mask_candidate=mask_candidate,
        )
        runtime_pool = None
        runtime_current = None
        if policy.schema_version == 3:
            ranked_edges = rank_scanner_candidates(
                loaded.normalized_bgr,
                edge_candidates,
                policy.parameters.exterior_config,
                selector_config=policy.parameters.selector_config,
            )
            edge_priors = {
                str(candidate["candidate_id"]): float(candidate["prior_score"])
                for candidate in edge_candidates
            }
            runtime_pool = [
                serialize_ranked_candidate(
                    candidate,
                    raw_prior_score=edge_priors[candidate.candidate_id],
                    analysis_source="edge",
                )
                for candidate in ranked_edges
            ]

            v7_seed_inputs = []
            for seed in seeds:
                full_corners = _v8_candidate_corners(seed)
                if full_corners is None:
                    continue
                stage_scores = seed.get("stage_scores", {})
                stage_scores = stage_scores if isinstance(stage_scores, dict) else {}
                raw_prior = stage_scores.get(
                    "pre_score", stage_scores.get("full_score", 0.0)
                )
                raw_sources = seed.get("sources", ())
                source_groups = (
                    (raw_sources,)
                    if isinstance(raw_sources, str)
                    else tuple(str(value) for value in raw_sources)
                )
                v7_seed_inputs.append({
                    "candidate_id": f"v7:audit:{seed['candidate_id']}",
                    "corners": map_full_normalized_to_analysis(
                        loaded, full_corners
                    ),
                    "prior_score": float(raw_prior),
                    "seed_candidate_ids": (str(seed["candidate_id"]),),
                    "seed_source_groups": source_groups,
                    "seed_support_count": 1,
                    "seed_source_group_count": max(1, len(source_groups)),
                })
            ranked_v7_seeds = rank_scanner_candidates(
                loaded.normalized_bgr,
                v7_seed_inputs,
                policy.parameters.exterior_config,
                selector_config=policy.parameters.selector_config,
            ) if v7_seed_inputs else ()
            seed_priors = {
                str(candidate["candidate_id"]): float(candidate["prior_score"])
                for candidate in v7_seed_inputs
            }
            runtime_pool.extend(
                serialize_ranked_candidate(
                    candidate,
                    raw_prior_score=seed_priors[candidate.candidate_id],
                    analysis_source="v7_seed",
                )
                for candidate in ranked_v7_seeds
            )

            if v7_candidate is not None:
                ranked_v7_current = rank_scanner_candidates(
                    loaded.normalized_bgr,
                    (v7_candidate,),
                    policy.parameters.exterior_config,
                    selector_config=policy.parameters.selector_config,
                )[0]
                runtime_pool.append(serialize_ranked_candidate(
                    ranked_v7_current,
                    raw_prior_score=float(v7_candidate["prior_score"]),
                    analysis_source="v7_current",
                ))
            if mask_candidate is not None:
                mask_prior = float(mask_candidate.get("score", 0.5))
                ranked_mask = rank_scanner_candidates(
                    loaded.normalized_bgr,
                    ({**dict(mask_candidate), "prior_score": mask_prior},),
                    policy.parameters.exterior_config,
                    selector_config=policy.parameters.selector_config,
                )[0]
                runtime_pool.append(serialize_ranked_candidate(
                    ranked_mask,
                    raw_prior_score=mask_prior,
                    analysis_source="mask",
                ))
            selected_id = selection.selected.candidate_id
            source_order = ("edge", "v7_current", "mask", "v7_seed")
            runtime_current = next(
                (
                    candidate
                    for source in source_order
                    for candidate in runtime_pool
                    if candidate["analysis_source"] == source
                    and candidate["candidate_id"] == selected_id
                ),
                None,
            )
            if runtime_current is None:
                raise ValueError("selected runtime candidate is absent from pool")
    except Exception:
        return _v8_fallback_result(
            core, mask_result.status.value, "candidate_or_selector_error"
        )

    if decision.status in {V8CascadeStatus.V7_FALLBACK, V8CascadeStatus.CANCELLED}:
        return _v8_fallback_result(
            core,
            mask_result.status.value,
            decision.reason,
            cancelled=decision.status is V8CascadeStatus.CANCELLED,
        )
    if policy.schema_version == 3:
        from dataclasses import replace

        try:
            boundary_result = evaluate_boundary_quality(
                loaded.normalized_bgr,
                pool=runtime_pool,
                current_candidate=runtime_current,
                mask_probability=mask_probability,
                image_size=tuple(loaded.normalized_size),
                decision_status=decision.status.value,
                decision_reason=decision.reason,
                artifact=boundary_quality_artifact,
            )
            boundary_evidence = {
                "artifact_sha256": boundary_quality_artifact.source_sha256,
                "action": boundary_result["action"],
                "adopted_proposal": boundary_result["adopted_proposal"],
                "relative_gate": boundary_result["relative_gate"],
                "boundary_evaluation_count": boundary_result[
                    "boundary_evaluation_count"
                ],
                "mask_inference_count": boundary_result["mask_inference_count"],
                "proposal_source_combination": boundary_result["proposal"].get(
                    "source_combination"
                ),
            }
            evidence = {**dict(decision.evidence), "boundary_quality": boundary_evidence}
            if boundary_result["adopted_proposal"]:
                decision = replace(
                    decision,
                    status=V8CascadeStatus.AUTOMATIC,
                    reason="v8.2_boundary_manual_rescue",
                    selected_candidate_id="v8.2:boundary_proposal",
                    selected_corners=tuple(
                        tuple(float(value) for value in point)
                        for point in boundary_result["selected_corners"]
                    ),
                    evidence=evidence,
                )
            else:
                decision = replace(decision, evidence=evidence)
        except Exception as exc:
            decision = replace(
                decision,
                evidence={
                    **dict(decision.evidence),
                    "boundary_quality": {
                        "artifact_sha256": (
                            boundary_quality_artifact.source_sha256
                        ),
                        "action": "keep_practical_v4",
                        "failure": type(exc).__name__,
                    },
                },
            )
    output = _v8_core_from_v7(core, loaded, decision, policy)
    return V8DormantResult(
        core_payload=output,
        audit_envelope={
            "requested_detector": "v8",
            "v8_provider_status": mask_result.status.value,
            "v8_fallback_reason": None,
            "v8_policy_sha256": policy.policy_sha256,
            "v8_decision_status": decision.status.value,
            "v8_selection_reason": decision.reason,
        },
        status=decision.status,
    )


def _auto_terminal_entry(img_path: str, relative_path: Optional[str], reason: str,
                         scene_profile: str = DEFAULT_SCENE_PROFILE) -> Dict[str, Any]:
    """Return a complete, persistable fail-closed auto result."""
    from photocut.algorithms.v7.cascade import CASCADE_POLICY_VERSION

    filename = relative_path or os.path.basename(img_path)
    return {
        "filename": filename,
        "algorithm_version": V7_ALGORITHM_VERSION,
        "algorithm_boundary_corners": [],
        "boundary_corners": [],
        "algorithm_corners": [],
        "algorithm_preview_corners": [],
        "corners": [],
        "preview_corners": [],
        "manual_corners": None,
        "manual_preview_corners": None,
        "crop_corners": None,
        "inset": None,
        "original_size": [0, 0],
        "preview_size": [0, 0],
        "confidences": [],
        "success": False,
        "confirmed": False,
        "manually_adjusted": False,
        "algorithm_generated": True,
        "adjust_count": 0,
        "adjust_timestamp": None,
        "detector": "manual_review",
        "detector_requested": "auto",
        "detector_used": "manual_review",
        "scene_profile": scene_profile,
        "detection_status": "manual_review",
        "error_message": reason,
        "fallback_reason": "input_decode_error",
        "cascade_policy_version": CASCADE_POLICY_VERSION,
        "cascade_calls": {"v7": 0, "v5.2": 0},
    }


def _map_analysis_entry_to_full(entry: Dict[str, Any], loaded: Any) -> Dict[str, Any]:
    """Map analysis-space detector output to full normalized source pixels."""
    analysis_size = tuple(int(v) for v in loaded.normalized_size)
    full_size = tuple(int(v) for v in loaded.full_normalized_size)
    if analysis_size != full_size:
        for field in (
            "algorithm_boundary_corners", "boundary_corners", "algorithm_corners",
            "corners", "alternate_corners", "v52_corners",
        ):
            points = entry.get(field)
            if isinstance(points, (tuple, list)) and len(points) == 4:
                try:
                    entry[field] = [list(point) for point in loaded.map_analysis_to_full(points)]
                except (TypeError, ValueError, IndexError):
                    entry[field] = []
        audits = entry.get("candidate_audit")
        if isinstance(audits, list):
            mapped_audits = []
            for raw_audit in audits:
                audit = dict(raw_audit) if isinstance(raw_audit, dict) else raw_audit
                if isinstance(audit, dict):
                    for field in (
                        "original_legal_corners", "pre_topk_corners",
                        "proposed_refined_corners", "adopted_refined_corners",
                    ):
                        points = audit.get(field)
                        if isinstance(points, (tuple, list)) and len(points) == 4:
                            try:
                                audit[field] = [list(point) for point in loaded.map_analysis_to_full(points)]
                            except (TypeError, ValueError, IndexError):
                                audit[field] = None
                mapped_audits.append(audit)
            entry["candidate_audit"] = mapped_audits

    corners = entry.get("corners") or entry.get("algorithm_boundary_corners") or []
    entry.setdefault("algorithm_boundary_corners", [list(point) for point in corners])
    entry.setdefault("boundary_corners", [list(point) for point in corners])
    entry.setdefault("algorithm_corners", [list(point) for point in corners])
    entry["original_size"] = [full_size[0], full_size[1]]
    entry["analysis_size"] = [analysis_size[0], analysis_size[1]]
    entry["normalized_size"] = [full_size[0], full_size[1]]

    if max(full_size) > PREVIEW_MAX_SIZE:
        preview_scale = PREVIEW_MAX_SIZE / float(max(full_size))
        preview_size = [
            max(1, int(round(full_size[0] * preview_scale))),
            max(1, int(round(full_size[1] * preview_scale))),
        ]
    else:
        preview_size = [full_size[0], full_size[1]]
    entry["preview_size"] = preview_size

    def preview_points(points: Any) -> list[list[int]]:
        if not isinstance(points, (tuple, list)) or len(points) != 4:
            return []
        return [[
            int(round(float(point[0]) * preview_size[0] / full_size[0])),
            int(round(float(point[1]) * preview_size[1] / full_size[1])),
        ] for point in points]

    entry["algorithm_preview_corners"] = preview_points(entry.get("algorithm_boundary_corners"))
    entry["preview_corners"] = preview_points(entry.get("boundary_corners"))
    return entry


def detect_and_save_corners_v84(
    img_path: str, output_dir: str, *, runtime: Any, loaded_input: Any,
    relative_path: Optional[str] = None, request_id: Optional[str] = None,
    image_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Project one V8.4 candidate; every new result requires human confirmation."""
    if loaded_input is None:
        raise ValueError("V8.4 requires a decoded, normalized input")
    prediction = runtime.predict(loaded_input)
    status = prediction["status"]
    points = prediction.get("analysis_corners")
    valid = status == "candidate_requires_confirmation" and points is not None
    corners = [[float(x), float(y)] for x, y in points] if valid else []
    identity = {
        "request_id": request_id or loaded_input.source_sha256,
        "image_id": image_id or "sha256:" + loaded_input.source_sha256,
        "orientation_transform": loaded_input.orientation_transform,
        "algorithm_version": "8.4", "model_sha256": runtime.model_sha256,
        "mode": "requires_confirmation",
    }
    entry = {
        "filename": relative_path or os.path.basename(img_path),
        "algorithm_version": "8.4", "detector": "v8.4",
        "detector_requested": "v8.4", "detector_used": "v8.4",
        "model_sha256": runtime.model_sha256, "v84_status": status,
        "detection_status": status, "detection_identity": identity,
        "detection_id": identity["request_id"],
        "detection_parameters": {"model_sha256": runtime.model_sha256},
        "scene_profile": "scanner_white", "source_sha256": loaded_input.source_sha256,
        "source_original_size": list(loaded_input.original_size),
        "normalized_orientation": loaded_input.orientation_transform,
        "success": valid, "confirmed": False, "requires_confirmation": True,
        "manually_adjusted": False, "algorithm_generated": True,
        "manual_corners": None, "manual_preview_corners": None,
        "crop_corners": None, "inset": None, "adjust_count": 0,
        "adjust_timestamp": None, "error_message": prediction.get("reason"),
        "confidences": [], "risks": ["需要人工确认"] if valid else [status],
        "candidate_sources": ["v8.4"], "alternate_corners": None,
        "candidate_audit": [], "detection_debug": {**prediction, "analysis_corners": corners if valid else None},
    }
    for field in ("algorithm_boundary_corners", "boundary_corners", "algorithm_corners", "corners"):
        entry[field] = [point[:] for point in corners]
    return _map_analysis_entry_to_full(entry, loaded_input)


def detect_and_save_corners_auto(
    img_path: str,
    output_dir: str,
    *,
    img: Optional[np.ndarray] = None,
    relative_path: Optional[str] = None,
    request_id: Optional[str] = None,
    image_id: Optional[str] = None,
    shrink_min: int = 25,
    shrink_max: int = 70,
    params: Optional[DetectionParameters] = None,
    cancellation_token: Any = None,
    loaded_input: Any = None,
    scene_profile: str = DEFAULT_SCENE_PROFILE,
) -> Dict[str, Any]:
    """Run one V7 pass and at most one bounded v5.2 cross-check."""
    import time
    from photocut.algorithms.v7.cascade import CascadeDecision, decide_auto
    from photocut.algorithms.v7.geometry import GeometryError, validate_quad
    from photocut.algorithms.v7.input import LoadedImage, decode_bytes_for_analysis, normalize_array
    from photocut.algorithms.v7.legacy_worker import SourceSnapshot, run_legacy_worker
    from photocut.algorithms.v7.parameters import V7Parameters

    started = time.monotonic()
    v7_params = V7Parameters(mode="safe", scene_profile=scene_profile)
    loaded = loaded_input
    if loaded is not None:
        if not isinstance(loaded, LoadedImage):
            raise TypeError("loaded_input must be a v7 LoadedImage")
    elif os.path.isfile(img_path):
        try:
            loaded = decode_bytes_for_analysis(
                Path(img_path).read_bytes(), max_pixels=v7_params.max_input_pixels,
                analysis_max_edge=4096,
                cancellation_token=cancellation_token,
            )
        except Exception as exc:
            return _auto_terminal_entry(img_path, relative_path, str(exc), scene_profile)
    elif img is not None:
        normalized = normalize_array(np.asarray(img))
        import hashlib
        digest = hashlib.sha256(normalized.tobytes()).hexdigest()
        loaded = LoadedImage(
            digest, str(normalized.dtype), int(normalized.shape[2]), 1, normalized,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (normalized.shape[1], normalized.shape[0]), (normalized.shape[1], normalized.shape[0]),
        )
    else:
        result = _auto_terminal_entry(img_path, relative_path, "input_missing", scene_profile)
        result["fallback_reason"] = "input_missing"
        return result

    snapshot = SourceSnapshot(
        loaded.normalized_bgr, loaded.source_sha256, loaded.original_size,
        loaded.normalized_size, loaded.orientation_transform,
    )
    v7_info = detect_and_save_corners_v7(
        img_path, output_dir, img=loaded.normalized_bgr, loaded_input=loaded,
        v7_mode="safe", relative_path=relative_path, request_id=request_id, image_id=image_id,
        scene_profile=scene_profile,
        allow_legacy_fallback=False,
    )
    full_size = tuple(int(v) for v in loaded.full_normalized_size)
    if tuple(v7_info.get("original_size") or ()) != full_size:
        v7_info = _map_analysis_entry_to_full(v7_info, loaded)
    width, height = full_size
    decision = decide_auto(v7_info, image_size=(width, height))
    legacy_info = None
    legacy_attempted = False
    if decision.needs_legacy:
        legacy_attempted = True
        remaining_s = max(0.0, 1.5 - (time.monotonic() - started))
        worker = run_legacy_worker(
            snapshot,
            timeout_s=remaining_s,
            cancellation_token=cancellation_token,
            shrink_min=shrink_min,
            shrink_max=shrink_max,
            params=params,
        )
        if worker.get("status") == "ok" and isinstance(worker.get("result"), dict):
            legacy_info = dict(worker["result"])
            legacy_info["filename"] = relative_path or os.path.basename(img_path)
            legacy_info = _map_analysis_entry_to_full(legacy_info, loaded)
            decision = decide_auto(v7_info, image_size=(width, height), legacy=legacy_info)
        elif worker.get("status") == "cancelled":
            decision = CascadeDecision("manual_review", "manual_review", reasons=("cancelled",))
        else:
            decision = CascadeDecision("manual_review", "manual_review",
                                       reasons=(str(worker.get("error", "legacy_worker_failed")),))

    if decision.detector_used == "v7":
        chosen = dict(decision.selected or v7_info)
    elif decision.detector_used == "v5.2" and legacy_info is not None:
        chosen = dict(legacy_info)
    else:
        chosen = dict(v7_info)
        if not v7_info.get("corners") and legacy_info is not None:
            try:
                validate_quad(legacy_info.get("corners"), image_size=(width, height))
            except (GeometryError, TypeError, ValueError):
                pass
            else:
                chosen = dict(legacy_info)
        chosen["confirmed"] = False
        chosen["detection_status"] = "manual_review"
        if legacy_info is not None and len(legacy_info.get("corners") or ()) == 4:
            chosen["v52_corners"] = [list(point) for point in legacy_info["corners"]]
    chosen["detector_requested"] = "auto"
    chosen["scene_profile"] = scene_profile
    chosen["detector_used"] = decision.detector_used
    chosen["detector"] = decision.detector_used
    chosen["fallback_reason"] = ";".join(decision.reasons) if decision.reasons else None
    from photocut.algorithms.v7.cascade import CASCADE_POLICY_VERSION
    chosen["cascade_policy_version"] = CASCADE_POLICY_VERSION
    chosen["cascade_calls"] = {"v7": 1, "v5.2": int(legacy_attempted)}
    chosen["cascade_v7_result"] = v7_info
    if legacy_info is not None:
        chosen["cascade_v52_result"] = legacy_info
    chosen["source_sha256"] = snapshot.source_sha256
    chosen["source_original_size"] = list(snapshot.original_size)
    chosen["normalized_orientation"] = snapshot.orientation_transform
    chosen.setdefault(
        "algorithm_version",
        V7_ALGORITHM_VERSION if decision.detector_used != "v5.2" else ALGORITHM_VERSION,
    )
    chosen.setdefault("confirmed", False)
    return chosen


def load_image(img_path: str) -> Optional[np.ndarray]:
    """
    加载图片，支持大图

    返回:
        img: numpy 数组，失败返回 None
    """
    try:
        img = cv2.imread(img_path)
        return img
    except Exception as e:
        print(f"加载图片失败 {img_path}: {e}")
        return None


def create_preview(img: np.ndarray, max_size: int = PREVIEW_MAX_SIZE) -> np.ndarray:
    """
    创建预览图（保持比例）

    返回:
        preview: 预览图
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def detect_and_save_corners(
    img_path: str,
    output_dir: str,
    shrink_min: int = 25,
    shrink_max: int = 70,
    img: Optional[np.ndarray] = None,
    params: DetectionParameters = DEFAULT_DETECTION_PARAMETERS,
    relative_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    检测图片四角并保存

    参数:
        img_path: 图片路径
        output_dir: 输出目录
        shrink_min: 最小收缩量
        shrink_max: 最大收缩量

    返回:
        info: 包含检测结果的字典
    """
    if img is None:
        img = load_image(img_path)
    params = params or DEFAULT_DETECTION_PARAMETERS
    if img is None:
        return {
            "filename": relative_path or os.path.basename(img_path),
            "success": False,
            "error_message": "Failed to load image"
        }

    h, w = img.shape[:2]
    preview = create_preview(img)
    preview_h, preview_w = preview.shape[:2]

    # 检测四角（原图坐标）
    started = time.perf_counter()
    try:
        corners, confidences, debug_info = detect_corners_detailed(
            img, shrink_min, shrink_max, params=params
        )
        success = True
        error_message = None
    except Exception as e:
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        confidences = []
        debug_info = []
        success = False
        error_message = str(e)
    duration_ms = (time.perf_counter() - started) * 1000

    # 计算预览图缩放比例
    scale_x = preview_w / w
    scale_y = preview_h / h
    preview_corners = [[int(x * scale_x), int(y * scale_y)] for x, y in corners]

    return {
        "filename": relative_path or os.path.basename(img_path),
        # 算法原始检测坐标（不可变，用于对比和重置）
        "algorithm_boundary_corners": [list(c) for c in corners],
        "boundary_corners": [list(c) for c in corners],
        "algorithm_version": ALGORITHM_VERSION,
        "detection_parameters": params.to_dict() if params else DEFAULT_DETECTION_PARAMETERS.to_dict(),
        "confidences": [float(confidence) for confidence in confidences],
        "detection_debug": debug_info,
        "detection_duration_ms": round(duration_ms, 3),
        "algorithm_corners": [list(c) for c in corners],  # 深拷贝
        "algorithm_preview_corners": [list(c) for c in preview_corners],
        # 用户调整后的坐标（初始为 None，表示未调整）
        "manual_corners": None,
        "manual_preview_corners": None,
        # 最终使用的坐标（初始指向算法坐标）
        "corners": [list(c) for c in corners],
        "preview_corners": [list(c) for c in preview_corners],
        "crop_corners": None,
        "inset": None,
        "original_size": [w, h],
        "preview_size": [preview_w, preview_h],
        "success": success,
        "manually_adjusted": False,
        "algorithm_generated": True,
        "confirmed": False,
        "error_message": error_message,
        # 调整统计
        "adjust_count": 0,
        "adjust_timestamp": None
    }


def crop_image(
    img_path: str,
    corners: List[List[int]],
    output_dir: str,
    relative_path: Optional[str] = None,
    img: Optional[np.ndarray] = None,
) -> Tuple[bool, str]:
    """
    裁剪图片

    重要说明：
    - corners 坐标应来自 detect_corners() 对原图的检测结果
    - 本函数使用原图进行裁剪，确保输出质量
    - 不要用 PS 色阶后的图进行裁剪

    参数:
        img_path: 图片路径
        corners: 四角坐标 [[x,y], [x,y], [x,y], [x,y]]
        output_dir: 输出目录

    返回:
        (success, output_path)
    """
    # 加载原图（不是色阶后的图）进行裁剪
    img = img if img is not None else load_image(img_path)
    if img is None:
        return False, ""

    try:
        warped = perspective_transform(img, corners)

        relative = Path(relative_path) if relative_path else Path(os.path.basename(img_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("crop relative path must stay below output directory")
        target_dir = Path(output_dir) / relative.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        target_info = target_dir.lstat()
        if target_dir.is_symlink() or not target_dir.is_dir():
            raise ValueError("crop output parent must be a regular directory")
        output_path = target_dir / f"{relative.stem}_裁切.jpg"

        # 保存（高质量 JPEG）
        cv2.imwrite(str(output_path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

        return True, str(output_path)

    except Exception as e:
        print(f"裁剪失败 {img_path}: {e}")
        return False, ""


def load_corners_info(json_path: str) -> List[Dict[str, Any]]:
    """
    加载 corners_info.json

    返回:
        corners_info: 列表
    """
    if not os.path.exists(json_path):
        return []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 兼容性处理：旧数据格式转换为新格式
    for entry in data:
        if "algorithm_corners" not in entry:
            # 旧格式：没有两套坐标
            corners = entry.get("corners", [])
            preview_corners = entry.get("preview_corners", corners)
            # 设置算法坐标为原始坐标
            entry["algorithm_corners"] = [list(c) for c in corners] if corners else []
            entry["algorithm_preview_corners"] = [list(c) for c in preview_corners] if preview_corners else []
            # 如果有手动调整，manual_corners 会与 algorithm_corners 不同
            if entry.get("manually_adjusted", False):
                entry["manual_corners"] = [list(c) for c in corners] if corners else []
                entry["manual_preview_corners"] = [list(c) for c in preview_corners] if preview_corners else []
            else:
                entry["manual_corners"] = None
                entry["manual_preview_corners"] = None

        boundary = entry.get("boundary_corners", entry.get("corners", []))
        entry.setdefault("boundary_corners", [list(c) for c in boundary])
        entry.setdefault(
            "algorithm_boundary_corners",
            [list(c) for c in entry.get("algorithm_corners", boundary)],
        )
        entry.setdefault("crop_corners", None)
        entry.setdefault("inset", None)

    return data


def save_corners_info(json_path: str, corners_info: List[Dict[str, Any]]) -> None:
    """
    保存 corners_info.json
    """
    atomic_write_json(Path(json_path), corners_info)


def apply_annotation_to_entry(entry: Dict[str, Any], event: Dict[str, Any]) -> None:
    """Apply the active durable annotation to its mutable UI cache entry."""
    boundary = [list(point) for point in event["boundary_corners"]]
    entry["boundary_corners"] = boundary
    entry["corners"] = [point[:] for point in boundary]
    entry["manual_corners"] = [point[:] for point in boundary]
    entry["confirmed"] = True
    entry["manually_adjusted"] = event["confirmation"] == "adjusted"
    entry["annotation_id"] = event["annotation_id"]
    entry["confirm_timestamp"] = event["confirmed_at"]
    entry.pop("draft_adjusted_corner_indices", None)


def reconcile_confirmation_events(
    corners_info: List[Dict[str, Any]], annotation_store: Any
) -> None:
    """Make the mutable confirmation cache reflect the latest durable events."""
    latest = annotation_store.latest_by_image()
    for entry in corners_info:
        event = latest.get(entry.get("image_id"))
        if event:
            apply_annotation_to_entry(entry, event)


def update_corners_entry(
    json_path: str,
    filename: str,
    update: Dict[str, Any]
) -> None:
    """
    更新指定文件的检测结果
    """
    corners_info = load_corners_info(json_path)

    for entry in corners_info:
        if entry["filename"] == filename:
            entry.update(update)
            break
    else:
        corners_info.append({
            "filename": filename,
            "corners": [[0, 0], [0, 0], [0, 0], [0, 0]],
            "original_size": [0, 0],
            "preview_size": [0, 0],
            "success": False,
            "manually_adjusted": False,
            "algorithm_generated": True,
            "confirmed": False,
            "error_message": "Not found in previous detection"
        })
        entry.update(update)

    save_corners_info(json_path, corners_info)
