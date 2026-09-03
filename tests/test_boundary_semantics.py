import unittest
from unittest.mock import patch

import numpy as np

from photocut.algorithms.v5_2 import detector as corner_detector


class BoundarySemanticsTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((200, 300, 3), dtype=np.uint8)
        self.raw = [[10, 20], [290, 20], [290, 180], [10, 180]]

    @patch("photocut.algorithms.v5_2.detector.detect_corners_v48_with_confidence")
    def test_high_confidence_path_returns_raw_boundary(self, detect_v48):
        detect_v48.return_value = (
            [point[:] for point in self.raw],
            [0.9, 0.9, 0.9, 0.9],
            [{}, {}, {}, {}],
        )

        actual = corner_detector.detect_corners_v410(self.image)

        self.assertEqual(self.raw, actual)

    @patch("photocut.algorithms.v5_2.detector.infer_by_edge_and_diagonal", return_value=(None, 0.0, None))
    @patch("photocut.algorithms.v5_2.detector.infer_by_two_edges", return_value=(None, 0.0, None))
    @patch("photocut.algorithms.v5_2.detector.detect_corners_v48_with_confidence")
    def test_fallback_path_does_not_add_second_inset(
        self, detect_v48, _infer_two, _infer_diagonal
    ):
        detect_v48.return_value = (
            [point[:] for point in self.raw],
            [0.1, 0.1, 0.1, 0.1],
            [{}, {}, {}, {}],
        )

        actual = corner_detector.detect_corners_v410(self.image)

        self.assertEqual(self.raw, actual)

    @patch("photocut.algorithms.v5_2.detector.infer_by_two_edges")
    @patch("photocut.algorithms.v5_2.detector.detect_corners_v48_with_confidence")
    def test_inferred_corner_is_not_inset(self, detect_v48, infer_two):
        raw = [[0, 0], [299, 0], [299, 199], [0, 199]]
        detect_v48.return_value = (
            [point[:] for point in raw],
            [0.9, 0.9, 0.9, 0.1],
            [{}, {}, {}, {}],
        )
        infer_two.return_value = ([0, 199], 0.8, 3)

        actual = corner_detector.detect_corners_v410(self.image)

        self.assertEqual(raw, actual)

    def test_boundary_clamp_allows_image_edge(self):
        self.assertEqual(
            [0, 199],
            corner_detector.clamp_corner_to_boundary([-4, 205], 300, 200, margin=0),
        )

    def test_boundary_helpers_treat_dimensions_as_exclusive(self):
        self.assertFalse(
            corner_detector.is_corner_in_boundary([300, 200], 300, 200)
        )
        self.assertGreater(
            corner_detector.calculate_boundary_violation([300, 200], 300, 200), 0
        )
        self.assertTrue(
            corner_detector.is_corner_in_boundary([294, 194], 300, 200, margin=5)
        )
        self.assertFalse(
            corner_detector.is_corner_in_boundary([295, 195], 300, 200, margin=5)
        )
        self.assertGreater(
            corner_detector.calculate_boundary_violation(
                [295, 195], 300, 200, margin=5
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
