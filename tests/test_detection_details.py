import time
import unittest
from unittest.mock import patch

import numpy as np

import photocut


class DetectionDetailsTests(unittest.TestCase):
    def test_each_call_uses_its_own_details_even_when_corners_are_identical(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        corners = [[1, 1], [18, 1], [18, 18], [1, 18]]
        first_details = (corners, [0.11, 0.12, 0.13, 0.14], [{"image": "first"}] * 4)
        second_details = (corners, [0.81, 0.82, 0.83, 0.84], [{"image": "second"}] * 4)

        with patch("photocut.core.detect_corners_detailed", side_effect=[first_details, second_details]) as detect:
            first = photocut.detect_and_save_corners("first.jpg", "unused", img=image)
            second = photocut.detect_and_save_corners("second.jpg", "unused", img=image)

        self.assertEqual(2, detect.call_count)
        self.assertEqual(first_details[1], first["confidences"])
        self.assertEqual(first_details[2], first["detection_debug"])
        self.assertEqual(second_details[1], second["confidences"])
        self.assertEqual(second_details[2], second["detection_debug"])

    def test_detector_failure_records_elapsed_time_from_the_same_call(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)

        def delayed_failure(*_args, **_kwargs):
            time.sleep(0.02)
            raise RuntimeError("delayed detector failure")

        with patch(
            "photocut.core.detect_corners_detailed", side_effect=delayed_failure
        ) as detect:
            result = photocut.detect_and_save_corners(
                "failed.jpg", "unused", img=image
            )

        detect.assert_called_once()
        self.assertFalse(result["success"])
        self.assertEqual("delayed detector failure", result["error_message"])
        self.assertGreaterEqual(result["detection_duration_ms"], 10.0)


if __name__ == "__main__":
    unittest.main()
