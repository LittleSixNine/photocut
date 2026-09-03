import argparse
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from photocut import cli as photocut_cli
import pytest


def _runtime():
    return SimpleNamespace(
        policy=SimpleNamespace(
            algorithm_version="8.1",
            policy_sha256="sha256:" + "1" * 64,
            validated_core_sha256="sha256:" + "2" * 64,
            model_id="sha256:" + "3" * 64,
            model_sha256="sha256:" + "4" * 64,
            parameters=SimpleNamespace(to_dict=lambda: {"schema_version": 2}),
        ),
        mask_provider=object(),
        runtime_config={
            "schema_version": 1,
            "provider": "CPUExecutionProvider",
            "execution_mode": "ORT_SEQUENTIAL",
            "intra_op_num_threads": 1,
            "inter_op_num_threads": 1,
        },
        runtime_config_sha256="sha256:" + "5" * 64,
        model_manifest_sha256="sha256:" + "6" * 64,
    )


def _selection_kwargs(runtime):
    return {
        "detector": "v8",
        "v7_mode": None,
        "scene_profile": "scanner_white",
        "img_path": Path("scan.jpg"),
        "output_dir": "out",
        "shrink_min": 25,
        "shrink_max": 70,
        "params": photocut_cli.DEFAULT_DETECTION_PARAMETERS,
        "v8_runtime": runtime,
    }


def test_v8_dispatch_uses_one_preloaded_runtime_and_returns_core_payload():
    runtime = _runtime()
    dormant = SimpleNamespace(
        core_payload={
            "filename": "scan.jpg",
            "algorithm_version": "8.1",
            "detector": "v8",
            "detector_requested": "v8",
            "detector_used": "v8",
            "success": True,
        },
        audit_envelope={
            "requested_detector": "v8",
            "v8_provider_status": "success",
            "v8_fallback_reason": None,
            "v8_policy_sha256": runtime.policy.policy_sha256,
            "v8_decision_status": "automatic",
            "v8_selection_reason": "edge_base",
        },
    )
    with patch(
        "photocut.cli.detect_and_save_corners_v8_dormant", return_value=dormant
    ) as adapter:
        info = photocut_cli._selected_detection(**_selection_kwargs(runtime))

    assert info["detector_used"] == "v8"
    assert info["v8_provider_status"] == "success"
    assert info["v8_selection_reason"] == "edge_base"
    assert adapter.call_args.kwargs["policy"] is runtime.policy
    assert adapter.call_args.kwargs["mask_provider"] is runtime.mask_provider


def test_v8_adapter_fallback_keeps_v7_geometry_but_records_v8_request():
    runtime = _runtime()
    corners = [[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]
    dormant = SimpleNamespace(
        core_payload={
            "filename": "scan.jpg",
            "algorithm_version": "7.1",
            "detector": "v7",
            "detector_requested": "v7",
            "detector_used": "v7",
            "algorithm_boundary_corners": corners,
            "corners": corners,
            "success": True,
        },
        audit_envelope={
            "requested_detector": "v8",
            "v8_provider_status": "error",
            "v8_fallback_reason": "mask_provider_error",
        },
    )
    with patch(
        "photocut.cli.detect_and_save_corners_v8_dormant", return_value=dormant
    ):
        info = photocut_cli._selected_detection(**_selection_kwargs(runtime))

    assert info["corners"] == corners
    assert info["algorithm_version"] == "7.1"
    assert info["detector_requested"] == "v8"
    assert info["detector_used"] == "v7"
    assert info["v8_fallback_reason"] == "mask_provider_error"


def test_v8_runtime_identity_is_resume_safe_and_uses_v8_algorithm_version():
    runtime = _runtime()
    args = argparse.Namespace(
        output="/tmp/output",
        shrink_min=25,
        shrink_max=70,
        detector="v8",
        v7_mode=None,
        scene_profile="scanner_white",
    )
    value = photocut_cli._runtime_for_detection(
        args, [], photocut_cli.DEFAULT_DETECTION_PARAMETERS, v8_runtime=runtime
    )

    assert value["algorithm_version"] == "8.1"
    parameters = value["parameters"]
    assert parameters["detector"] == "v8"
    assert parameters["v8_policy_sha256"] == runtime.policy.policy_sha256
    assert parameters["v8_validated_core_sha256"] == runtime.policy.validated_core_sha256
    assert parameters["v8_model_sha256"] == runtime.policy.model_sha256
    assert parameters["v8_runtime_config_sha256"] == runtime.runtime_config_sha256


def test_v8_entries_use_verified_normalized_snapshot_and_persist_identity():
    info = {
        "source_id": "source-1",
        "image_id": "sha256:" + "a" * 64,
        "batch_id": "batch-1",
        "run_id": "run-1",
        "algorithm_version": "8.1",
        "algorithm_boundary_corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
        "boundary_corners": [[1, 1], [9, 1], [9, 9], [1, 9]],
        "success": True,
        "detector": "v8",
        "detector_requested": "v8",
        "detector_used": "manual_review",
        "v8_policy_sha256": "sha256:" + "1" * 64,
        "v8_provider_status": "success",
        "v8_selection_reason": "strong_three_way_conflict",
    }

    assert photocut_cli._entry_uses_normalized_snapshot(info)
    persisted = photocut_cli._result_from_info(info)
    assert persisted["detector_requested"] == "v8"
    assert persisted["detector_used"] == "manual_review"
    assert persisted["v8_policy_sha256"] == info["v8_policy_sha256"]
    assert persisted["v8_selection_reason"] == info["v8_selection_reason"]


def test_v8_decode_failure_keeps_v8_identity_and_requires_manual_review():
    runtime = _runtime()
    info = photocut_cli._decode_failure_info(
        Path("broken.jpg"),
        "source-1",
        "sha256:" + "a" * 64,
        "batch-1",
        "run-1",
        detector="v8",
        v8_runtime=runtime,
    )

    assert info["algorithm_version"] == "8.1"
    assert info["detector_requested"] == "v8"
    assert info["detector_used"] == "manual_review"
    assert info["detection_identity"]["parameter_sha256"] == "1" * 64
    assert info["detection_parameters"] == {"schema_version": 2}
    assert info["v8_provider_status"] == "not_run"
    assert info["v8_fallback_reason"] == "input_decode_error"


def test_v8_runtime_is_required_for_v8_dispatch():
    with patch("photocut.cli.detect_and_save_corners_v8_dormant") as adapter:
        try:
            photocut_cli._selected_detection(**_selection_kwargs(None))
        except RuntimeError as exc:
            assert "V8" in str(exc)
        else:
            raise AssertionError("missing V8 runtime must fail before detection")
    adapter.assert_not_called()


def test_main_forwards_opt_in_v8_and_local_asset_overrides():
    with patch.object(
        sys,
        "argv",
        [
            "photocut_cli.py",
            "input",
            "--detect",
            "--detector",
            "v8",
            "--v8-policy",
            "policy.json",
            "--v8-model-dir",
            "model",
        ],
    ), patch("photocut.cli.detect_command") as command:
        photocut_cli.main()

    args = command.call_args.args[0]
    assert args.detector == "v8"
    assert args.v8_policy == "policy.json"
    assert args.v8_model_dir == "model"


def test_main_rejects_v8_on_non_scanner_profile():
    with patch.object(
        sys,
        "argv",
        [
            "photocut_cli.py",
            "input",
            "--detect",
            "--detector",
            "v8",
            "--scene-profile",
            "generic_single",
        ],
    ), pytest.raises(SystemExit):
        photocut_cli.main()


def test_detect_command_loads_one_v8_runtime_for_the_batch():
    runtime = _runtime()
    info = {
        "filename": "scan.jpg",
        "algorithm_version": "8.1",
        "detector": "v8",
        "detector_requested": "v8",
        "detector_used": "v8",
        "success": True,
        "confirmed": False,
    }
    with tempfile.TemporaryDirectory() as directory:
        loaded = object()
        args = argparse.Namespace(
            input=directory,
            output=str(Path(directory) / "output"),
            shrink_min=25,
            shrink_max=70,
            threshold=None,
            detector="v8",
            v7_mode=None,
            scene_profile=None,
            v8_policy=None,
            v8_model_dir=None,
            no_dataset_archive=True,
        )
        with patch(
            "photocut.cli.find_images",
            return_value=[(str(Path(directory) / "scan.jpg"), "scan.jpg")],
        ), patch(
            "photocut.cli._v8_runtime_from_args", return_value=runtime
        ) as load_runtime, patch(
            "photocut.cli._prepare_detection_input",
            return_value=(None, loaded, (100, 100)),
        ), patch(
            "photocut.cli._selected_detection", return_value=info
        ) as selected:
            photocut_cli.detect_command(args)

    load_runtime.assert_called_once_with(args)
    assert selected.call_count == 1
    assert selected.call_args.kwargs["v8_runtime"] is runtime
    assert selected.call_args.kwargs["loaded_input"] is loaded
