import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import photocut
from photocut import cli as photocut_cli
from photocut.algorithms.v7.input import LoadedImage
from photocut.algorithms.v7.types import ProviderResult, ProviderStatus
from photocut.algorithms.v8.code_seal import compute_validated_core_sha256
from photocut.algorithms.v8.boundary_quality import load_boundary_quality_artifact
from photocut.algorithms.v8.edge_hypotheses import EdgeHypothesisConfig, EdgeHypothesisResult, EdgeQuadHypothesis
from photocut.algorithms.v8.parameters import V8Parameters
from photocut.algorithms.v8.policy import V8Policy


PROJECT_ROOT = Path(photocut.__file__).resolve().parent
MODEL_ID = "sha256:" + "1" * 64
MODEL_SHA = "sha256:" + "2" * 64
MANIFEST_SHA = "sha256:" + "3" * 64
POPULATION_SHA = "sha256:" + "4" * 64
CALIBRATION_SHA = "sha256:" + "5" * 64
BOUNDARY_SHA = "sha256:4c91ffdac9a85c337a1a8a869deb41c9b598b2344388f61a1000a4bc31fdab86"
RUNTIME_CONFIG = {
    "schema_version": 1,
    "provider": "CPUExecutionProvider",
    "execution_mode": "ORT_SEQUENTIAL",
    "intra_op_num_threads": 1,
    "inter_op_num_threads": 1,
}


def _hash_json(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _loaded():
    image = np.full((100, 120, 3), 250, np.uint8)
    image[10:90, 15:105] = (80, 120, 180)
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return LoadedImage(
        "9" * 64,
        "uint8",
        3,
        1,
        image,
        identity,
        identity,
        (120, 100),
        (120, 100),
        (120, 100),
        identity,
    )


def _policy(*, core_sha=None, runtime_sha=None):
    return V8Policy(
        model_id=MODEL_ID,
        model_sha256=MODEL_SHA,
        model_manifest_sha256=MANIFEST_SHA,
        runtime_config_sha256=runtime_sha or _hash_json(RUNTIME_CONFIG),
        calibration_population_sha256=POPULATION_SHA,
        calibration_run_sha256=CALIBRATION_SHA,
        validated_core_sha256=core_sha or compute_validated_core_sha256(PROJECT_ROOT),
        parameters=V8Parameters(),
    )


def _policy_v82(*, core_sha=None):
    return V8Policy(
        model_id=MODEL_ID,
        model_sha256=MODEL_SHA,
        model_manifest_sha256=MANIFEST_SHA,
        runtime_config_sha256=_hash_json(RUNTIME_CONFIG),
        calibration_population_sha256=POPULATION_SHA,
        calibration_run_sha256=CALIBRATION_SHA,
        validated_core_sha256=core_sha or compute_validated_core_sha256(PROJECT_ROOT),
        boundary_quality_artifact_sha256=BOUNDARY_SHA,
        parameters=V8Parameters(),
        schema_version=3,
        policy_id="v8-auto-v3",
        algorithm_version="8.2",
    )


def _v7_core():
    corners = [[15.0, 10.0], [104.0, 10.0], [104.0, 89.0], [15.0, 89.0]]
    return {
        "filename": "scan.jpg",
        "algorithm_version": "7.1",
        "algorithm_boundary_corners": corners,
        "boundary_corners": corners,
        "algorithm_corners": corners,
        "corners": corners,
        "algorithm_preview_corners": [[15, 10], [104, 10], [104, 89], [15, 89]],
        "preview_corners": [[15, 10], [104, 10], [104, 89], [15, 89]],
        "original_size": [120, 100],
        "preview_size": [120, 100],
        "success": True,
        "confirmed": False,
        "detector": "v7",
        "detector_requested": "v7",
        "detector_used": "v7",
        "detection_status": "v7_recommended",
        "detection_id": "request-1",
        "detection_identity": {
            "request_id": "request-1",
            "image_id": "image-1",
            "orientation_transform": "exif_1",
            "algorithm_version": "7.1",
            "parameter_sha256": "a" * 64,
            "mode": "safe",
        },
        "candidate_audit": [
            {
                "candidate_id": "seed-1",
                "sources": ["background:min_area_rect"],
                "stage_ranks": {"selected": 1},
                "stage_scores": {"final": 0.9},
                "adopted_refined_corners": corners,
                "original_legal_corners": corners,
            }
        ],
        "legacy_fallback": False,
        "fallback_reason": None,
    }


class Provider:
    def __init__(self, result):
        self.result = result
        self.calls = 0
        self.threshold = 0.2
        self.manifest = SimpleNamespace(
            model_id=MODEL_ID,
            model_sha256=MODEL_SHA,
            manifest_sha256=MANIFEST_SHA,
        )

    def provide(self, context, params, cancellation_token=None, deadline=None):
        self.calls += 1
        assert params.scene_profile == "scanner_white"
        return self.result

    def provide_with_probability(
        self, context, params, cancellation_token=None, deadline=None
    ):
        result = self.provide(context, params, cancellation_token, deadline)
        return result, np.full(context.image.shape[:2], 0.5, np.float32)


def _provider_result(status, candidates=(), *, diagnostics=None):
    return ProviderResult(
        "photo_mask_v8",
        candidates,
        status,
        diagnostics=diagnostics or {},
    )


def test_model_timeout_returns_byte_for_byte_equivalent_v7_core_payload():
    expected = _v7_core()
    provider = Provider(_provider_result(ProviderStatus.TIMEOUT))
    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected) as v7:
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
        )

    assert result.core_payload == expected
    assert result.audit_envelope == {
        "requested_detector": "v8",
        "v8_provider_status": "timeout",
        "v8_fallback_reason": "mask_provider_timeout",
    }
    assert provider.calls == 1
    v7.assert_called_once()


def test_v82_retries_one_transient_v7_error_before_running_mask_provider():
    failed = _v7_core()
    failed.update({
        "algorithm_boundary_corners": [],
        "boundary_corners": [],
        "algorithm_corners": [],
        "corners": [],
        "algorithm_preview_corners": [],
        "preview_corners": [],
        "candidate_audit": [],
        "success": False,
        "detection_status": "error",
    })
    recovered = _v7_core()
    provider = Provider(_provider_result(ProviderStatus.TIMEOUT))
    artifact = load_boundary_quality_artifact(
        PROJECT_ROOT / "algorithms/v8/assets/v8.2-boundary-quality-v6.json"
    )

    with patch(
        "photocut.core.detect_and_save_corners_v7",
        side_effect=(failed, recovered),
    ) as v7:
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy_v82(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            boundary_quality_artifact=artifact,
            loaded_input=_loaded(),
        )

    assert result.core_payload == recovered
    assert result.audit_envelope["v8_fallback_reason"] == "mask_provider_timeout"
    assert provider.calls == 1
    assert v7.call_count == 2


def test_missing_model_provider_falls_back_without_changing_v7_identity():
    expected = _v7_core()
    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected) as v7:
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy(),
            mask_provider=None,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
        )
    assert result.core_payload == expected
    assert result.audit_envelope["v8_fallback_reason"] == "mask_provider_unavailable"
    v7.assert_called_once()


def test_core_or_runtime_identity_mismatch_never_runs_model_and_falls_back():
    expected = _v7_core()
    provider = Provider(_provider_result(ProviderStatus.SUCCESS))
    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected):
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy(core_sha="sha256:" + "0" * 64),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
        )
    assert result.core_payload == expected
    assert result.audit_envelope["v8_fallback_reason"] == "validated_core_mismatch"
    assert provider.calls == 0


def test_success_path_runs_v7_model_and_edge_generation_once_and_emits_v8_identity():
    expected = _v7_core()
    mask_corners = ((15.0, 10.0), (104.0, 10.0), (104.0, 89.0), (15.0, 89.0))
    mask_candidate = {
        "candidate_id": "mask-1",
        "corners": mask_corners,
        "score": 0.9,
        "sources": ("photo_mask_v8",),
        "evidence": {
            "foreground_probability": 0.95,
            "main_component_ratio": 0.98,
            "polygon_mask_iou": 0.94,
            "boundary_entropy": 0.4,
            "visible_margin": 0.01,
            "refinement": {"max_normalized_shift": 0.0},
        },
    }
    provider = Provider(_provider_result(ProviderStatus.SUCCESS, (mask_candidate,)))
    generated = EdgeHypothesisResult(
        candidates=(
            EdgeQuadHypothesis("edge-1", mask_corners, 100.0, (0.0, 0.0, 0.0, 0.0)),
        ),
        edge_evidence=(),
        config=EdgeHypothesisConfig(),
    )
    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected) as v7, patch(
        "photocut.algorithms.v8.edge_hypotheses.generate_edge_hypotheses", return_value=generated
    ) as edge:
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
        )

    assert result.core_payload["algorithm_version"] == "8.1"
    assert result.core_payload["detector"] == "v8"
    assert result.core_payload["v8_policy_sha256"] == _policy().policy_sha256
    assert _policy().policy_id == "v8-auto-v2"
    assert result.core_payload["corners"] == [list(point) for point in mask_corners]
    assert result.audit_envelope["v8_provider_status"] == "success"
    assert provider.calls == 1
    v7.assert_called_once()
    edge.assert_called_once()


def test_v82_requires_bound_artifact_before_running_model():
    expected = _v7_core()
    provider = Provider(_provider_result(ProviderStatus.SUCCESS))
    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected):
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy_v82(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
        )

    assert result.core_payload == expected
    assert result.audit_envelope["v8_fallback_reason"] == (
        "boundary_quality_artifact_mismatch"
    )
    assert provider.calls == 0


def test_v82_full_frame_mask_probability_falls_back_before_selection():
    expected = _v7_core()
    provider = Provider(_provider_result(
        ProviderStatus.NO_CANDIDATE,
        diagnostics={"reason": "mask_candidate_refinement_invalid"},
    ))
    artifact = load_boundary_quality_artifact(
        PROJECT_ROOT / "algorithms/v8/assets/v8.2-boundary-quality-v6.json",
        expected_sha256=BOUNDARY_SHA,
    )

    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected):
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy_v82(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
            boundary_quality_artifact=artifact,
        )

    assert result.core_payload == expected
    assert result.audit_envelope == {
        "requested_detector": "v8",
        "v8_provider_status": "no_candidate",
        "v8_fallback_reason": "mask_candidate_refinement_invalid",
    }
    assert provider.calls == 1


def test_v82_adopts_a_bound_manual_rescue_after_practical_decision():
    expected = _v7_core()
    corners = ((15.0, 10.0), (104.0, 10.0), (104.0, 89.0), (15.0, 89.0))
    rescued = ((16.0, 11.0), (103.0, 11.0), (103.0, 88.0), (16.0, 88.0))
    mask_candidate = {
        "candidate_id": "mask-1",
        "corners": corners,
        "score": 0.9,
        "sources": ("photo_mask_v8",),
        "evidence": {
            "foreground_probability": 0.95,
            "main_component_ratio": 0.98,
            "polygon_mask_iou": 0.94,
            "boundary_entropy": 0.4,
            "visible_margin": 0.01,
            "refinement": {"max_normalized_shift": 0.0},
        },
    }
    provider = Provider(
        _provider_result(ProviderStatus.SUCCESS, (mask_candidate,))
    )
    generated = EdgeHypothesisResult(
        candidates=(
            EdgeQuadHypothesis(
                "edge-1", corners, 100.0, (0.0, 0.0, 0.0, 0.0)
            ),
        ),
        edge_evidence=(),
        config=EdgeHypothesisConfig(),
    )
    artifact = load_boundary_quality_artifact(
        PROJECT_ROOT / "algorithms/v8/assets/v8.2-boundary-quality-v6.json",
        expected_sha256=BOUNDARY_SHA,
    )
    boundary_result = {
        "action": "rescue_manual",
        "adopted_proposal": True,
        "selected_corners": rescued,
        "proposal": {"source_combination": "edge+edge+mask+mask"},
        "relative_gate": {"action": "rescue_manual"},
        "boundary_evaluation_count": 48,
        "mask_inference_count": 1,
    }

    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected), patch(
        "photocut.algorithms.v8.edge_hypotheses.generate_edge_hypotheses", return_value=generated
    ), patch(
        "photocut.algorithms.v8.boundary_quality.evaluate_boundary_quality",
        return_value=boundary_result,
    ):
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy_v82(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
            boundary_quality_artifact=artifact,
        )

    assert result.status.value == "automatic"
    assert result.core_payload["algorithm_version"] == "8.2"
    assert result.core_payload["corners"] == [list(point) for point in rescued]
    boundary = result.core_payload["v8_cascade_evidence"]["boundary_quality"]
    assert boundary["action"] == "rescue_manual"
    assert boundary["mask_inference_count"] == 1


def test_success_path_preserves_duplicate_edge_geometry_seed_support():
    expected = _v7_core()
    second_seed = dict(expected["candidate_audit"][0])
    second_seed.update({
        "candidate_id": "seed-2",
        "sources": ["background:border_connected"],
        "stage_ranks": {"selected": 2},
    })
    expected["candidate_audit"] = [expected["candidate_audit"][0], second_seed]
    corners = ((15.0, 10.0), (104.0, 10.0), (104.0, 89.0), (15.0, 89.0))
    mask_candidate = {
        "candidate_id": "mask-1",
        "corners": corners,
        "score": 0.9,
        "sources": ("photo_mask_v8",),
        "evidence": {
            "foreground_probability": 0.95,
            "main_component_ratio": 0.98,
            "polygon_mask_iou": 0.94,
            "boundary_entropy": 0.4,
            "visible_margin": 0.01,
            "refinement": {"max_normalized_shift": 0.0},
        },
    }
    provider = Provider(_provider_result(ProviderStatus.SUCCESS, (mask_candidate,)))
    generated = EdgeHypothesisResult(
        candidates=(
            EdgeQuadHypothesis("edge-shared", corners, 100.0, (0.0, 0.0, 0.0, 0.0)),
        ),
        edge_evidence=(),
        config=EdgeHypothesisConfig(),
    )

    with patch("photocut.core.detect_and_save_corners_v7", return_value=expected), patch(
        "photocut.algorithms.v8.edge_hypotheses.generate_edge_hypotheses", return_value=generated
    ) as edge:
        result = photocut.detect_and_save_corners_v8_dormant(
            "scan.jpg",
            "unused",
            policy=_policy(),
            mask_provider=provider,
            runtime_config=RUNTIME_CONFIG,
            loaded_input=_loaded(),
        )

    cascade_evidence = result.core_payload["detection_debug"]["v8_cascade"]["evidence"]
    assert cascade_evidence["selected_seed_support_count"] == 2
    assert cascade_evidence["selected_seed_source_group_count"] == 2
    assert edge.call_count == 2


def test_generic_profile_is_rejected_before_v7_or_model_runs():
    provider = Provider(_provider_result(ProviderStatus.SUCCESS))
    with patch("photocut.core.detect_and_save_corners_v7") as v7:
        with pytest.raises(ValueError, match="scanner_white"):
            photocut.detect_and_save_corners_v8_dormant(
                "scan.jpg",
                "unused",
                policy=_policy(),
                mask_provider=provider,
                runtime_config=RUNTIME_CONFIG,
                loaded_input=_loaded(),
                scene_profile="generic_single",
            )
    assert provider.calls == 0
    v7.assert_not_called()


def test_cli_exposes_opt_in_v8_while_default_remains_auto():
    parser = argparse.ArgumentParser()
    photocut_cli.add_detector_arguments(parser, default="auto")
    detector = next(action for action in parser._actions if action.dest == "detector")
    assert tuple(detector.choices) == ("auto", "v5.2", "v7", "v8")
    assert photocut_cli.DEFAULT_DETECTOR == "auto"
