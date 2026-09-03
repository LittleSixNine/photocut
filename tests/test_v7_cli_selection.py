import argparse
import json
import sys
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

import photocut
from photocut import cli as photocut_cli
from photocut.config import ALGORITHM_VERSION, V7_ALGORITHM_VERSION
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.types import DetectionIdentity, DetectionResult, DetectionStatus


class V7CliSelectionTests(unittest.TestCase):
    def test_archived_auto_input_skips_legacy_full_resolution_loader(self):
        loaded = SimpleNamespace(full_normalized_size=(17095, 11475))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as source, patch(
            "photocut.algorithms.v7.input.decode_bytes_for_analysis", return_value=loaded
        ) as bounded, patch("photocut.cli.load_image") as legacy_load:
            image, prepared, image_size = photocut_cli._prepare_detection_input(
                "auto", Path(source.name), v7_mode=None
            )

        self.assertIsNone(image)
        self.assertIs(prepared, loaded)
        self.assertEqual((17095, 11475), image_size)
        bounded.assert_called_once()
        legacy_load.assert_not_called()

    def test_explicit_v7_bounded_input_projects_to_full_coordinates(self):
        from photocut.algorithms.v7.input import LoadedImage

        preview = np.zeros((50, 100, 3), dtype=np.uint8)
        loaded = LoadedImage(
            "b" * 64, "uint8", 3, 1, preview,
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (400, 200), (100, 50), (400, 200),
            ((399.0 / 99.0, 0.0, 0.0), (0.0, 199.0 / 49.0, 0.0), (0.0, 0.0, 1.0)),
        )
        result = self._result(
            DetectionStatus.V7_RECOMMENDED,
            ((0.0, 0.0), (99.0, 0.0), (99.0, 49.0), (0.0, 49.0)),
        )
        with patch("photocut.algorithms.v7.detector.detect_corners_v7", return_value=result):
            info = photocut.detect_and_save_corners_v7(
                "missing.jpg", "unused", loaded_input=loaded,
            )
        self.assertEqual(
            [[0.0, 0.0], [399.0, 0.0], [399.0, 199.0], [0.0, 199.0]],
            info["corners"],
        )
        self.assertEqual([400, 200], info["normalized_size"])
        self.assertEqual([100, 50], info["analysis_size"])

    def test_archived_auto_decode_failure_keeps_cascade_identity(self):
        info = photocut_cli._decode_failure_info(
            Path("broken.jpg"), "source-1", "sha256:" + "a" * 64,
            "batch-1", "run-1", detector="auto",
        )
        self.assertEqual("auto", info["detector_requested"])
        self.assertEqual("manual_review", info["detector_used"])
        self.assertEqual("v7-auto-v3", info["cascade_policy_version"])
        self.assertEqual({"v7": 0, "v5.2": 0}, info["cascade_calls"])
        self.assertRegex(info["cascade_policy_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(info["v5_2_parameter_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(info["v5_2_code_identity"], r"^[0-9a-f]{64}$")
        self.assertEqual("a" * 64, info["source_sha256"])

        persisted = photocut_cli._result_from_info(info)
        self.assertEqual("auto", persisted["detector_requested"])
        self.assertEqual("manual_review", persisted["detector_used"])
        self.assertEqual({"v7": 0, "v5.2": 0}, persisted["cascade_calls"])

    @staticmethod
    def _result(status, corners=None, *, mode="safe", error=None, risks=(), orientation="identity"):
        params = V7Parameters(mode=mode)
        identity = DetectionIdentity(
            "request-1", "image-1", orientation, V7_ALGORITHM_VERSION, params.sha256(), mode
        )
        return DetectionResult(
            identity=identity, status=status, corners=corners,
            alternate_corners=corners, overall_confidence=.91 if corners else None,
            edge_confidences=(.91, .90, .89, .88) if corners else (),
            corner_confidences=(.91, .90, .89, .88) if corners else (),
            risks=risks, error=error,
        )

    def test_omitted_detector_uses_auto_v7_first_default(self):
        with patch.object(sys, "argv", ["photocut_cli.py", "input", "--detect"]), patch(
            "photocut.cli.detect_command"
        ) as command:
            photocut_cli.main()
        args = command.call_args.args[0]
        self.assertEqual("auto", args.detector)
        self.assertIsNone(args.v7_mode)
        self.assertIsNone(args.scene_profile)

    def test_v7_detector_and_mode_are_forwarded(self):
        with patch.object(
            sys,
            "argv",
            ["photocut_cli.py", "input", "--detect", "--detector", "v7", "--v7-mode", "aggressive"],
        ), patch("photocut.cli.detect_command") as command:
            photocut_cli.main()
        args = command.call_args.args[0]
        self.assertEqual("v7", args.detector)
        self.assertEqual("aggressive", args.v7_mode)

    def test_scene_profile_is_forwarded_independently_from_detector_mode(self):
        with patch.object(
            sys,
            "argv",
            ["photocut_cli.py", "input", "--detect", "--scene-profile", "generic_single"],
        ), patch("photocut.cli.detect_command") as command:
            photocut_cli.main()

        args = command.call_args.args[0]
        self.assertEqual("auto", args.detector)
        self.assertEqual("generic_single", args.scene_profile)

    def test_auto_dispatch_uses_cascade_adapter(self):
        info = {"detector": "v7", "detector_used": "v7", "success": True}
        with patch("photocut.cli.detect_and_save_corners_auto", return_value=info) as auto, patch(
            "photocut.cli.detect_and_save_corners_v7"
        ) as v7:
            result = photocut_cli._selected_detection(
                detector="auto", v7_mode=None, scene_profile="scanner_white",
                img_path=Path("scan.jpg"), output_dir="out",
                shrink_min=25, shrink_max=70, params=photocut_cli.DEFAULT_DETECTION_PARAMETERS,
            )
        self.assertEqual("v7", result["detector_used"])
        auto.assert_called_once()
        self.assertEqual("scanner_white", auto.call_args.kwargs["scene_profile"])
        v7.assert_not_called()

    def test_v7_mode_without_v7_is_rejected(self):
        with patch.object(
            sys, "argv", ["photocut_cli.py", "input", "--detect", "--v7-mode", "safe"]
        ), self.assertRaises(SystemExit):
            photocut_cli.main()

    def test_detector_options_work_after_detect_subcommand(self):
        with patch.object(
            sys,
            "argv",
            ["photocut_cli.py", "input", "detect", "--detector", "v7", "--v7-mode", "safe"],
        ), patch("photocut.cli.detect_command") as command:
            photocut_cli.main()
        args = command.call_args.args[0]
        self.assertEqual(("v7", "safe"), (args.detector, args.v7_mode))

    def test_v7_runtime_identity_contains_mode_and_parameter_hash(self):
        args = argparse.Namespace(
            output="/tmp/output",
            shrink_min=25,
            shrink_max=70,
            detector="v7",
            v7_mode="safe",
            scene_profile="generic_single",
        )
        runtime = photocut_cli._runtime_for_detection(args, [], photocut_cli.DEFAULT_DETECTION_PARAMETERS)
        params = runtime["parameters"]
        self.assertEqual("v7", params["detector"])
        self.assertEqual("safe", params["v7_mode"])
        self.assertEqual("generic_single", params["scene_profile"])
        self.assertEqual("generic_single", params["v7_parameters"]["scene_profile"])
        self.assertEqual(64, len(params["v7_parameter_sha256"]))
        self.assertEqual(V7_ALGORITHM_VERSION, params["v7_algorithm_version"])

    def test_auto_runtime_identity_contains_policy_and_legacy_code_hash(self):
        args = argparse.Namespace(
            output="/tmp/output", shrink_min=25, shrink_max=70,
            detector="auto", v7_mode=None,
            scene_profile="scanner_white",
        )
        policy = photocut_cli._load_auto_v4_policy(
            Path(photocut_cli.__file__).resolve().parent
            / photocut_cli.AUTO_V4_POLICY_RELATIVE_PATH
        )
        auto_runtime = photocut_cli.AutoV4Runtime(
            policy, "explicit", "v7", "v7", False, None, None
        )
        runtime = photocut_cli._runtime_for_detection(
            args,
            [],
            photocut_cli.DEFAULT_DETECTION_PARAMETERS,
            auto_runtime=auto_runtime,
        )
        params = runtime["parameters"]
        self.assertEqual("auto", params["detector_requested"])
        self.assertEqual("scanner_white", params["scene_profile"])
        self.assertEqual("auto-v4", params["auto_cascade_version"])
        self.assertEqual("v7-auto-v3", params["auto_v3_policy_version"])
        self.assertEqual(64, len(params["auto_v3_policy_sha256"]))
        self.assertEqual(64, len(params["v5_2_code_identity"]))

    def test_projection_preserves_recommended_polygon_and_identity(self):
        corners = ((2.25, 3.5), (17.0, 3.5), (17.0, 16.0), (2.25, 16.0))
        result = self._result(DetectionStatus.V7_RECOMMENDED, corners)
        with patch("photocut.algorithms.v7.detector.detect_corners_v7", return_value=result):
            info = photocut.detect_and_save_corners_v7(
                "missing.jpg", "unused", img=np.zeros((20, 20, 3), dtype=np.uint8),
                v7_mode="safe", relative_path="scan.jpg",
            )
        self.assertTrue(info["success"])
        self.assertFalse(info["confirmed"])
        self.assertEqual("v7_recommended", info["detection_status"])
        self.assertEqual("request-1", info["detection_id"])
        self.assertEqual([[2, 4], [17, 4], [17, 16], [2, 16]], info["algorithm_preview_corners"])
        self.assertEqual("v7", info["detector"])

    def test_projection_fallback_keeps_actual_corners_and_reason(self):
        corners = ((1.0, 1.0), (18.0, 1.0), (18.0, 18.0), (1.0, 18.0))
        result = self._result(
            DetectionStatus.V52_FALLBACK, corners,
            risks=("zero_legal_candidates",), error=None,
        )
        with patch("photocut.algorithms.v7.detector.detect_corners_v7", return_value=result):
            info = photocut.detect_and_save_corners_v7(
                "missing.jpg", "unused", img=np.zeros((20, 20, 3), dtype=np.uint8),
            )
        self.assertTrue(info["success"])
        self.assertTrue(info["legacy_fallback"])
        self.assertEqual("zero_legal_candidates", info["fallback_reason"])
        self.assertEqual([[1.0, 1.0], [18.0, 1.0], [18.0, 18.0], [1.0, 18.0]], info["corners"])

    def test_projection_no_primary_is_explicitly_uncroppable(self):
        result = self._result(
            DetectionStatus.NO_PRIMARY_PHOTO,
            error="multiple_primary_ambiguity",
            risks=("multiple_primary_ambiguity",),
        )
        with patch("photocut.algorithms.v7.detector.detect_corners_v7", return_value=result):
            info = photocut.detect_and_save_corners_v7(
                "missing.jpg", "unused", img=np.zeros((20, 20, 3), dtype=np.uint8),
            )
        self.assertFalse(info["success"])
        self.assertFalse(info["confirmed"])
        self.assertEqual([], info["corners"])
        self.assertEqual("no_primary_photo", info["detection_status"])

    def test_v7_dispatch_is_used_only_when_selected(self):
        info = {"filename": "scan.jpg", "success": True}
        with patch("photocut.cli.detect_and_save_corners_v7", return_value=info) as v7, patch(
            "photocut.cli.detect_and_save_corners", return_value=info
        ) as v52:
            photocut_cli._selected_detection(
                detector="v7", v7_mode="safe", scene_profile="generic_single",
                img_path=Path("scan.jpg"), output_dir="out",
                shrink_min=25, shrink_max=70, params=photocut_cli.DEFAULT_DETECTION_PARAMETERS,
            )
            photocut_cli._selected_detection(
                detector="v5.2", v7_mode=None, scene_profile=None,
                img_path=Path("scan.jpg"), output_dir="out",
                shrink_min=25, shrink_max=70, params=photocut_cli.DEFAULT_DETECTION_PARAMETERS,
            )
        v7.assert_called_once()
        self.assertEqual("generic_single", v7.call_args.kwargs["scene_profile"])
        v52.assert_called_once()

    def test_file_adapter_uses_v7_exif_normalized_input(self):
        image = Image.new("RGB", (8, 4), (240, 20, 20))
        exif = image.getexif()
        exif[274] = 6  # 90-degree clockwise; normalized shape becomes 4x8.
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle:
            image.save(handle.name, exif=exif)
            captured = {}
            def fake_detect(value, **kwargs):
                captured["value"] = value
                return self._result(DetectionStatus.NO_PRIMARY_PHOTO, error="no_photo", orientation="exif_6")
            with patch("photocut.algorithms.v7.detector.detect_corners_v7", side_effect=fake_detect):
                info = photocut.detect_and_save_corners_v7(handle.name, "unused")
        self.assertEqual("exif_6", info["normalized_orientation"])
        self.assertEqual((8, 4, 3), captured["value"].normalized_bgr.shape)

    def test_file_adapter_fails_closed_without_legacy_full_decode(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg") as handle, patch(
            "photocut.algorithms.v7.input.decode_bytes_for_analysis", side_effect=ValueError("too large")
        ) as bounded, patch("photocut.core.load_image") as legacy_load:
            info = photocut.detect_and_save_corners_v7(handle.name, "unused")

        self.assertFalse(info["success"])
        bounded.assert_called_once()
        legacy_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
