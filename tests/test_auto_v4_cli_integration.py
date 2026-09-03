import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from photocut import cli as photocut_cli
from photocut import core as photocut_core
from photocut.data.dataset_store import DatasetStore


def _args(**overrides):
    values = {
        "detector": "auto",
        "auto_engine": None,
        "scene_profile": "scanner_white",
        "v8_policy": None,
        "v8_model_dir": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class AutoV4CliIntegrationTests(unittest.TestCase):
    def test_production_v8_observer_counts_calls_in_packaged_core(self):
        result = SimpleNamespace(audit_envelope={})

        def adapter(*_args, **_kwargs):
            photocut_core.detect_and_save_corners_v7("scan.jpg", "output")
            return result

        with patch.object(
            photocut_cli, "detect_and_save_corners_v8_dormant", adapter
        ), patch.object(
            photocut_core, "detect_and_save_corners_v8_dormant", adapter
        ), patch.object(
            photocut_core, "detect_and_save_corners_v7", return_value={}
        ) as v7:
            observed_result, calls = photocut_cli._call_v8_adapter_observed()

        self.assertIs(result, observed_result)
        self.assertEqual(1, calls)
        v7.assert_called_once_with("scan.jpg", "output")

    def test_cli_accepts_explicit_auto_engine(self):
        with patch.object(
            sys,
            "argv",
            ["photocut_cli.py", "input", "--detect", "--auto-engine", "v7"],
        ), patch("photocut.cli.detect_command") as command:
            photocut_cli.main()

        self.assertEqual("auto", command.call_args.args[0].detector)
        self.assertEqual("v7", command.call_args.args[0].auto_engine)

    def test_cli_rejects_auto_engine_with_explicit_detector(self):
        with patch.object(
            sys,
            "argv",
            [
                "photocut_cli.py", "input", "--detect", "--detector", "v7",
                "--auto-engine", "v7",
            ],
        ), self.assertRaises(SystemExit):
            photocut_cli.main()

    def test_cli_rejects_v8_auto_engine_for_generic_scene(self):
        with patch.object(
            sys,
            "argv",
            [
                "photocut_cli.py", "input", "--detect", "--scene-profile",
                "generic_single", "--auto-engine", "v8",
            ],
        ), self.assertRaises(SystemExit):
            photocut_cli.main()

    def test_new_implicit_scanner_batch_uses_v8_when_preflight_passes(self):
        runtime = SimpleNamespace(policy=SimpleNamespace(algorithm_version="8.2"))
        with patch("photocut.cli._v8_runtime_from_args", return_value=runtime):
            resolved = photocut_cli._resolve_auto_v4_runtime(_args())

        self.assertEqual("implicit_default", resolved.request_mode)
        self.assertEqual("v8", resolved.requested_engine)
        self.assertEqual("v8", resolved.effective_engine)
        self.assertFalse(resolved.degraded)
        self.assertIs(runtime, resolved.v8_runtime)

    def test_new_implicit_scanner_batch_degrades_whole_batch_on_preflight_failure(self):
        with patch(
            "photocut.cli._v8_runtime_from_args", side_effect=RuntimeError("missing")
        ):
            resolved = photocut_cli._resolve_auto_v4_runtime(_args())

        self.assertEqual("v7", resolved.effective_engine)
        self.assertTrue(resolved.degraded)
        self.assertEqual("v8_runtime_unavailable", resolved.degradation_reason)
        self.assertIsNone(resolved.v8_runtime)

    def test_new_explicit_v8_preflight_failure_is_fatal(self):
        with patch(
            "photocut.cli._v8_runtime_from_args", side_effect=RuntimeError("missing")
        ), self.assertRaisesRegex(RuntimeError, "V8"):
            photocut_cli._resolve_auto_v4_runtime(_args(auto_engine="v8"))

    def test_existing_route_inherits_implicit_requested_and_effective_engine(self):
        existing = {
            "auto_engine_request_mode": "implicit_default",
            "auto_engine_requested": "v8",
            "auto_engine_effective": "v7",
            "auto_engine_degraded": True,
            "auto_engine_degradation_reason": "v8_runtime_unavailable",
        }
        with patch("photocut.cli._v8_runtime_from_args") as load:
            resolved = photocut_cli._resolve_auto_v4_runtime(
                _args(), existing_route=existing
            )

        self.assertEqual("implicit_default", resolved.request_mode)
        self.assertEqual("v8", resolved.requested_engine)
        self.assertEqual("v7", resolved.effective_engine)
        load.assert_not_called()

    def test_existing_route_rejects_request_mode_or_engine_mismatch(self):
        existing = {
            "auto_engine_request_mode": "implicit_default",
            "auto_engine_requested": "v8",
            "auto_engine_effective": "v7",
            "auto_engine_degraded": True,
            "auto_engine_degradation_reason": "v8_runtime_unavailable",
        }
        with self.assertRaisesRegex(RuntimeError, "request mode"):
            photocut_cli._resolve_auto_v4_runtime(
                _args(auto_engine="v7"), existing_route=existing
            )

    def test_existing_effective_v8_never_degrades_during_resume(self):
        existing = {
            "auto_engine_request_mode": "explicit",
            "auto_engine_requested": "v8",
            "auto_engine_effective": "v8",
            "auto_engine_degraded": False,
            "auto_engine_degradation_reason": None,
        }
        with patch(
            "photocut.cli._v8_runtime_from_args", side_effect=RuntimeError("missing")
        ), self.assertRaisesRegex(RuntimeError, "resume"):
            photocut_cli._resolve_auto_v4_runtime(
                _args(auto_engine="v8"), existing_route=existing
            )

    def test_generic_scene_is_fixed_to_v7_without_v8_preflight(self):
        with patch("photocut.cli._v8_runtime_from_args") as load:
            resolved = photocut_cli._resolve_auto_v4_runtime(
                _args(scene_profile="generic_single")
            )
        self.assertEqual("v7", resolved.requested_engine)
        self.assertEqual("v7", resolved.effective_engine)
        load.assert_not_called()

    def test_existing_route_is_selected_by_output_and_ordered_inputs(self):
        sources = [{"source_id": "source-1", "image_id": "sha256:1", "source_filename": "a.jpg", "source_path": "/a.jpg"}]
        with tempfile.TemporaryDirectory() as directory:
            store = DatasetStore(Path(directory).resolve() / "dataset")
            store.start_batch(runtime={
                "algorithm_version": "auto-v4",
                "parameters": {
                    "output_identity": "/tmp/output",
                    "input_images": sources,
                    "detector_requested": "auto",
                    "auto_cascade_version": "auto-v4",
                    "auto_engine_request_mode": "implicit_default",
                    "auto_engine_requested": "v8",
                    "auto_engine_effective": "v7",
                    "auto_engine_degraded": True,
                    "auto_engine_degradation_reason": "v8_runtime_unavailable",
                },
            })

            route = photocut_cli._existing_auto_v4_route(
                store,
                output_identity="/tmp/output",
                source_records=sources,
                allow_final_repair=True,
            )

        self.assertEqual("v7", route["auto_engine_effective"])
        self.assertEqual("implicit_default", route["auto_engine_request_mode"])

    def test_auto_v4_runtime_identity_binds_wrapper_and_both_detectors(self):
        policy = photocut_cli._load_auto_v4_policy(
            Path(photocut_cli.__file__).resolve().parent
            / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH
        )
        v8_policy = SimpleNamespace(
            algorithm_version="8.2",
            policy_sha256="sha256:" + "1" * 64,
            validated_core_sha256="sha256:" + "2" * 64,
            model_id="model-1",
            model_sha256="sha256:" + "3" * 64,
            parameters=SimpleNamespace(to_dict=lambda: {"schema_version": 3}),
            boundary_quality_artifact_sha256="sha256:" + "4" * 64,
        )
        v8_runtime = SimpleNamespace(
            policy=v8_policy,
            model_manifest_sha256="sha256:" + "5" * 64,
            runtime_config_sha256="sha256:" + "6" * 64,
        )
        auto_runtime = photocut_cli.AutoV4Runtime(
            policy, "implicit_default", "v8", "v8", False, None, v8_runtime
        )
        args = argparse.Namespace(
            output="/tmp/output", shrink_min=25, shrink_max=70,
            detector="auto", v7_mode=None, scene_profile="scanner_white",
            auto_engine=None,
        )

        runtime = photocut_cli._runtime_for_detection(
            args, [], photocut_cli.DEFAULT_DETECTION_PARAMETERS,
            auto_runtime=auto_runtime,
        )

        self.assertEqual("auto-v4", runtime["algorithm_version"])
        parameters = runtime["parameters"]
        self.assertEqual("auto-v4", parameters["auto_cascade_version"])
        self.assertEqual("PhotoCut Selector v4", parameters["selector_name"])
        self.assertEqual("4.0", parameters["selector_version"])
        self.assertEqual(policy.policy_sha256, parameters["auto_cascade_policy_sha256"])
        self.assertEqual("v8", parameters["auto_engine_effective"])
        self.assertRegex(parameters["auto_v4_wrapper_sha256"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(v8_policy.policy_sha256, parameters["v8_policy_sha256"])
        self.assertRegex(parameters["v5_2_code_identity"], r"^[0-9a-f]{64}$")

    def test_selected_detection_routes_effective_v8_to_auto_wrapper(self):
        policy = photocut_cli._load_auto_v4_policy(
            Path(photocut_cli.__file__).resolve().parent
            / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH
        )
        runtime = photocut_cli.AutoV4Runtime(
            policy, "implicit_default", "v8", "v8", False, None, object()
        )
        loaded = object()
        expected = {"detector_requested": "auto", "detector_used": "v8"}
        with patch(
            "photocut.cli._detect_auto_v4_v8", return_value=expected
        ) as wrapper, patch("photocut.cli.detect_and_save_corners_auto") as old_auto:
            result = photocut_cli._selected_detection(
                detector="auto", v7_mode=None, scene_profile="scanner_white",
                img_path=Path("scan.jpg"), output_dir="out",
                shrink_min=25, shrink_max=70,
                params=photocut_cli.DEFAULT_DETECTION_PARAMETERS,
                loaded_input=loaded, auto_runtime=runtime,
            )

        self.assertIs(expected, result)
        wrapper.assert_called_once()
        self.assertIs(loaded, wrapper.call_args.kwargs["loaded_input"])
        old_auto.assert_not_called()

    def test_selected_detection_routes_effective_v7_to_exact_auto_v3(self):
        policy = photocut_cli._load_auto_v4_policy(
            Path(photocut_cli.__file__).resolve().parent
            / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH
        )
        runtime = photocut_cli.AutoV4Runtime(
            policy, "explicit", "v7", "v7", False, None, None
        )
        expected = {"detector_requested": "auto", "detector_used": "v7"}
        with patch(
            "photocut.cli.detect_and_save_corners_auto", return_value=expected
        ) as old_auto, patch("photocut.cli._detect_auto_v4_v8") as wrapper:
            result = photocut_cli._selected_detection(
                detector="auto", v7_mode=None, scene_profile="scanner_white",
                img_path=Path("scan.jpg"), output_dir="out",
                shrink_min=25, shrink_max=70,
                params=photocut_cli.DEFAULT_DETECTION_PARAMETERS,
                loaded_input=object(), auto_runtime=runtime,
            )

        self.assertIs(expected, result)
        old_auto.assert_called_once()
        wrapper.assert_not_called()

    def test_no_archive_freezes_one_auto_runtime_and_passes_one_predecode(self):
        policy = photocut_cli._load_auto_v4_policy(
            Path(photocut_cli.__file__).resolve().parent
            / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH
        )
        active_v8 = object()
        runtime = photocut_cli.AutoV4Runtime(
            policy, "implicit_default", "v8", "v8", False, None, active_v8
        )
        loaded = object()
        info = {"filename": "scan.jpg", "success": True, "confirmed": False}
        with tempfile.TemporaryDirectory() as directory:
            input_dir = Path(directory) / "input"
            input_dir.mkdir()
            source = input_dir / "scan.jpg"
            source.write_bytes(b"source")
            args = argparse.Namespace(
                input=str(input_dir),
                output=str(Path(directory) / "out"),
                shrink_min=25,
                shrink_max=70,
                threshold=None,
                detector="auto",
                auto_engine=None,
                v7_mode=None,
                scene_profile="scanner_white",
                v8_policy=None,
                v8_model_dir=None,
                no_dataset_archive=True,
            )
            with patch(
                "photocut.cli._resolve_auto_v4_runtime", return_value=runtime
            ) as resolve, patch(
                "photocut.cli._prepare_detection_input",
                return_value=(None, loaded, (100, 100)),
            ) as prepare, patch(
                "photocut.cli._selected_detection", return_value=info
            ) as selected:
                photocut_cli.detect_command(args)

        resolve.assert_called_once_with(args)
        prepare.assert_called_once()
        self.assertIs(runtime, selected.call_args.kwargs["auto_runtime"])
        self.assertIs(active_v8, selected.call_args.kwargs["v8_runtime"])
        self.assertIs(loaded, selected.call_args.kwargs["loaded_input"])

    def test_auto_v4_decode_failure_is_manual_and_preserves_route_identity(self):
        policy = photocut_cli._load_auto_v4_policy(
            Path(photocut_cli.__file__).resolve().parent
            / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH
        )
        v8 = SimpleNamespace(
            policy=SimpleNamespace(policy_sha256="sha256:" + "1" * 64)
        )
        runtime = photocut_cli.AutoV4Runtime(
            policy, "implicit_default", "v8", "v8", False, None, v8
        )

        info = photocut_cli._decode_failure_info(
            Path("broken.jpg"),
            "source-1",
            "sha256:" + "a" * 64,
            "batch-1",
            "run-1",
            detector="auto",
            auto_runtime=runtime,
        )
        persisted = photocut_cli._result_from_info(info)

        self.assertEqual("manual_review", info["detector_used"])
        self.assertEqual("error", info["auto_v4_status"])
        self.assertEqual({"v7": 0, "v8": 0, "v5.2": 0}, info["cascade_calls"])
        self.assertEqual("auto-v4", persisted["auto_cascade_version"])
        self.assertEqual("v8", persisted["auto_engine_effective"])


if __name__ == "__main__":
    unittest.main()
