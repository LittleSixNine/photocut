import unittest

from photocut.geometry.crop import InsetGeometryError, inset_quadrilateral


class InsetQuadrilateralTests(unittest.TestCase):
    def assertPointsAlmostEqual(self, expected, actual, places=3):
        for expected_point, actual_point in zip(expected, actual):
            for expected_value, actual_value in zip(expected_point, actual_point):
                self.assertAlmostEqual(expected_value, actual_value, places=places)

    def test_axis_aligned_rectangle_moves_each_edge_exactly_five_pixels(self):
        actual = inset_quadrilateral(
            [[10, 10], [110, 10], [110, 70], [10, 70]],
            distance_px=5,
            image_size=(120, 80),
        )
        self.assertPointsAlmostEqual(
            [[15, 15], [105, 15], [105, 65], [15, 65]], actual
        )

    def test_rotated_square_uses_edge_normals_not_xy_offsets(self):
        actual = inset_quadrilateral(
            [[50, 10], [90, 50], [50, 90], [10, 50]],
            distance_px=5,
            image_size=(100, 100),
        )
        self.assertPointsAlmostEqual(
            [[50, 17.071], [82.929, 50], [50, 82.929], [17.071, 50]], actual
        )

    def test_zero_distance_returns_original_boundary(self):
        corners = [[10, 10], [110, 10], [110, 70], [10, 70]]
        self.assertPointsAlmostEqual(
            corners, inset_quadrilateral(corners, 0, (120, 80))
        )

    def test_rejects_non_convex_quadrilateral(self):
        with self.assertRaisesRegex(InsetGeometryError, "convex"):
            inset_quadrilateral(
                [[10, 10], [110, 10], [30, 30], [10, 70]], 5, (120, 80)
            )

    def test_rejects_inset_that_collapses_polygon(self):
        with self.assertRaises(InsetGeometryError):
            inset_quadrilateral(
                [[10, 10], [20, 10], [20, 20], [10, 20]], 6, (30, 30)
            )


if __name__ == "__main__":
    unittest.main()
