"""Tests for DetectionParameters dataclass and parameter propagation."""
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np


class DetectionParametersTests(unittest.TestCase):
    def test_defaults_match_current_v52_configuration(self):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS, DetectionParameters
        params = DEFAULT_DETECTION_PARAMETERS
        self.assertEqual(230, params.ps_highlight)
        self.assertEqual(5, params.gaussian_blur_ksize)
        self.assertEqual((30, 90), (params.canny_low, params.canny_high))
        self.assertEqual((35, 40, 25), (
            params.hough_threshold,
            params.hough_min_line_length,
            params.hough_max_line_gap,
        ))
        self.assertEqual((0.5, 0.3, 0.2), (
            params.weight_proximity, params.weight_length, params.weight_angle
        ))
        self.assertEqual((10.0, 10.0), (
            params.horizontal_angle_tolerance, params.vertical_angle_tolerance
        ))
        self.assertEqual((0.10, 0.15, 0.20, 0.25), params.roi_scales)

    def test_rejects_invalid_threshold_and_weights(self):
        from photocut.detection_parameters import DetectionParameters
        with self.assertRaises(ValueError):
            DetectionParameters(canny_low=100, canny_high=90)
        with self.assertRaises(ValueError):
            DetectionParameters(
                weight_proximity=0.5, weight_length=0.5, weight_angle=0.5
            )

    def test_json_round_trip_normalizes_roi_scales(self):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS, DetectionParameters

        restored = DetectionParameters.from_dict(
            json.loads(json.dumps(DEFAULT_DETECTION_PARAMETERS.to_dict()))
        )

        self.assertEqual(DEFAULT_DETECTION_PARAMETERS, restored)
        self.assertIsInstance(restored.roi_scales, tuple)

    @patch("photocut.algorithms.v5_2.detector.detect_corners_v410")
    def test_public_detector_forwards_exact_parameter_object(self, detect_v410):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        from photocut.algorithms.v5_2 import detector as corner_detector
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        params = DEFAULT_DETECTION_PARAMETERS.replace(ps_highlight=220)
        detect_v410.return_value = ([[1, 1], [18, 1], [18, 18], [1, 18]], [1.0]*4, [{}, {}, {}, {}])
        corner_detector.detect_corners(image, params=params)
        self.assertIs(params, detect_v410.call_args.kwargs["params"])


class DetectionParameterPropagationTests(unittest.TestCase):
    @patch("photocut.algorithms.v5_2.detector.ps_levels", side_effect=lambda image, **_kwargs: image)
    @patch("photocut.algorithms.v5_2.detector.detect_corner_in_roi")
    def test_roi_fallback_reuses_last_configured_scale(self, detect, _ps_levels):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        from photocut.algorithms.v5_2 import detector as corner_detector

        seen_shapes = []

        def no_corner(roi, *_args, **_kwargs):
            seen_shapes.append(roi.shape[:2])
            return (0, 0), 0.0, {"horizontal_lines": 0, "vertical_lines": 0}

        detect.side_effect = no_corner
        params = DEFAULT_DETECTION_PARAMETERS.replace(roi_scales=(0.20, 0.30))
        corner_detector.detect_corners_detailed(
            np.zeros((100, 100, 3), np.uint8), params=params
        )

        self.assertEqual([(20, 20), (30, 30), (30, 30)] * 4, seen_shapes)

    @patch("photocut.algorithms.v5_2.detector.ps_levels", side_effect=lambda image, **_kwargs: image)
    @patch("photocut.algorithms.v5_2.detector.detect_corner_in_roi")
    def test_legacy_roi_fallback_reuses_last_configured_scale(self, detect, _ps_levels):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        from photocut.algorithms.v5_2 import detector as corner_detector

        seen_shapes = []

        def no_corner(roi, *_args, **_kwargs):
            seen_shapes.append(roi.shape[:2])
            return (0, 0), 0.0, {"horizontal_lines": 0, "vertical_lines": 0}

        detect.side_effect = no_corner
        params = DEFAULT_DETECTION_PARAMETERS.replace(roi_scales=(0.20, 0.30))
        corner_detector.detect_corners(
            np.zeros((100, 100, 3), np.uint8), params=params
        )

        self.assertEqual([(20, 20), (30, 30), (30, 30)] * 4, seen_shapes)

    @patch("photocut.algorithms.v5_2.detector.cv2.Canny", return_value=np.zeros((20, 20), np.uint8))
    @patch("photocut.algorithms.v5_2.detector.cv2.HoughLinesP", return_value=None)
    def test_roi_detector_uses_custom_canny_thresholds(self, _hough, canny):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        from photocut.algorithms.v5_2 import detector as corner_detector

        params = DEFAULT_DETECTION_PARAMETERS.replace(canny_low=7, canny_high=19)
        corner_detector.detect_corner_in_roi(
            np.zeros((20, 20, 3), np.uint8), 0, params=params
        )

        self.assertEqual((7, 19), canny.call_args.args[1:3])

    def test_line_score_uses_custom_weights(self):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        from photocut.algorithms.v5_2 import detector as corner_detector

        line = np.array([0, 50, 100, 50])
        proximity_only = DEFAULT_DETECTION_PARAMETERS.replace(
            weight_proximity=1.0, weight_length=0.0, weight_angle=0.0
        )
        length_only = DEFAULT_DETECTION_PARAMETERS.replace(
            weight_proximity=0.0, weight_length=1.0, weight_angle=0.0
        )

        first = corner_detector.classify_line(line, 200, 100, proximity_only)[1]
        second = corner_detector.classify_line(line, 200, 100, length_only)[1]

        self.assertNotEqual(first, second)

    @patch("photocut.core.detect_corners_detailed")
    def test_production_detection_passes_the_recorded_parameter_object(self, detect):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        import photocut

        detect.return_value = ([[0, 0], [19, 0], [19, 19], [0, 19]], [1.0] * 4, [])
        params = DEFAULT_DETECTION_PARAMETERS.replace(canny_low=7, canny_high=19)
        info = photocut.detect_and_save_corners(
            "scan.jpg", "unused", img=np.zeros((20, 20, 3), np.uint8), params=params
        )

        self.assertIs(params, detect.call_args.kwargs["params"])
        self.assertEqual(params.to_dict(), info["detection_parameters"])

    @patch("photocut.algorithms.v5_2.detector.ps_levels")
    @patch("photocut.algorithms.v5_2.detector.detect_corner_in_roi")
    def test_legacy_highlight_replaces_parameter_without_mutating_caller(self, detect, ps_levels):
        from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS
        from photocut.algorithms.v5_2 import detector as corner_detector

        detect.return_value = ((0, 0), 0.0, {"horizontal_lines": 0, "vertical_lines": 0})
        ps_levels.side_effect = lambda image, highlight: image
        params = DEFAULT_DETECTION_PARAMETERS.replace(ps_highlight=217)
        image = np.zeros((20, 20, 3), np.uint8)
        corner_detector.detect_corners_detailed(image, highlight=211, params=params)

        self.assertEqual(211, ps_levels.call_args.kwargs["highlight"])
        self.assertEqual(217, params.ps_highlight)


class CliParameterCompatibilityTests(unittest.TestCase):
    @patch.object(sys, "argv", ["photocut_cli.py", "input", "--detect"])
    @patch("photocut.cli.detect_command")
    def test_default_cli_namespace_preserves_legacy_threshold(self, detect_command):
        from photocut import cli as photocut_cli
        photocut_cli.main()

        args = detect_command.call_args.args[0]
        self.assertEqual(240, args.threshold)
        self.assertEqual(
            240, photocut_cli.effective_detection_parameters(args.threshold).ps_highlight
        )
