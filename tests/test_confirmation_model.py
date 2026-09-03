import math
import unittest

import numpy as np

from photocut.confirmation.model import (
    ConfirmationState,
    DisplayTransform,
    MagnifierDragCapture,
    build_annotation_event,
    display_to_original,
    legacy_preview_corners,
    magnifier_drag_delta,
    move_corner,
    original_to_display,
    render_magnifier_source,
)
from photocut.confirmation.pointer import PointerReadError, PointerSample


class MagnifierDragCaptureTests(unittest.TestCase):
    def test_capture_moves_outside_window_and_stops_on_release(self):
        capture = MagnifierDragCapture()
        capture.begin(PointerSample(100.0, 100.0, True))

        self.assertEqual((-10, -5), capture.update(PointerSample(140.0, 120.0, True), zoom=4))
        self.assertTrue(capture.active)
        self.assertEqual((0, 0), capture.update(PointerSample(600.0, 300.0, False), zoom=4))
        self.assertFalse(capture.active)
        self.assertEqual((0, 0), capture.update(PointerSample(700.0, 300.0, True), zoom=4))

    def test_capture_preserves_signed_residual_across_small_moves(self):
        capture = MagnifierDragCapture()
        capture.begin(PointerSample(100.0, 100.0, True))

        first = capture.update(PointerSample(105.0, 100.0, True), zoom=4)
        second = capture.update(PointerSample(108.0, 100.0, True), zoom=4)
        self.assertEqual((-1, 0), first)
        self.assertEqual((-1, 0), second)

        capture.cancel()
        capture.begin(PointerSample(100.0, 100.0, True))
        self.assertEqual((-2, 0), capture.update(PointerSample(106.0, 100.0, True), zoom=4))
        self.assertEqual((0, 0), capture.update(PointerSample(108.0, 100.0, True), zoom=4))

    def test_capture_cancel_is_idempotent_and_invalid_sample_stops(self):
        capture = MagnifierDragCapture()
        capture.begin(PointerSample(1.0, 1.0, True))
        with self.assertRaises(PointerReadError):
            capture.update(PointerSample(float("nan"), 1.0, True), zoom=4)
        self.assertFalse(capture.active)
        capture.cancel()
        capture.cancel()


class ConfirmationStateTests(unittest.TestCase):
    def setUp(self):
        self.algorithm_corners = [[10, 10], [90, 10], [90, 70], [10, 70]]
        self.state = ConfirmationState(
            algorithm_corners=[point[:] for point in self.algorithm_corners],
            work_corners=[point[:] for point in self.algorithm_corners],
            image_size=(100, 80),
        )

    def test_returning_corner_to_algorithm_position_clears_adjustment(self):
        self.state.selected = 2

        self.state.move_selected(-1, 1)
        self.assertEqual({2}, self.state.adjusted_corner_indices)

        self.state.move_selected(1, -1)
        self.assertEqual(set(), self.state.adjusted_corner_indices)

    def test_reset_restores_algorithm_points_and_clears_adjustments(self):
        self.state.selected = 0
        self.state.move_selected(3, 4)

        self.state.reset()

        self.assertEqual(self.algorithm_corners, self.state.work_corners)
        self.assertEqual(set(), self.state.adjusted_corner_indices)
        self.assertIsNot(self.state.work_corners, self.state.algorithm_corners)
        for work, algorithm in zip(
            self.state.work_corners, self.state.algorithm_corners
        ):
            self.assertIsNot(work, algorithm)

    def test_rejects_invalid_selection_and_zoom(self):
        for selected in (-1, 4):
            with self.subTest(selected=selected):
                self.state.selected = selected
                with self.assertRaisesRegex(ValueError, "select a corner"):
                    self.state.move_selected(1, 0)

        for zoom in (1, 3, 32):
            with self.subTest(zoom=zoom):
                with self.assertRaisesRegex(ValueError, "zoom must be one of"):
                    self.state.set_zoom(zoom)

        for zoom in (2, 4, 8, 16):
            with self.subTest(zoom=zoom):
                self.state.set_zoom(zoom)
                self.assertEqual(zoom, self.state.zoom)

    def test_persisted_draft_reenter_restore_and_confirm_is_accepted(self):
        draft = [point[:] for point in self.algorithm_corners]
        draft[1][0] += 2
        state = ConfirmationState(
            algorithm_corners=[point[:] for point in self.algorithm_corners],
            work_corners=draft,
            image_size=(100, 80),
        )

        self.assertEqual({1}, state.adjusted_corner_indices)
        state.selected = 1
        state.move_selected(-2, 0)
        event = build_annotation_event(
            "image", "run", state.algorithm_corners, state.work_corners,
            state.adjusted_corner_indices, 0,
        )

        self.assertEqual("accepted", event["confirmation"])
        self.assertEqual([], event["adjusted_corner_indices"])


class LegacyPreviewMetadataTests(unittest.TestCase):
    def test_uses_independent_truncated_axis_scales(self):
        corners = [[16319, 10430], [16804, 11428], [1, 1], [8000, 5000]]

        preview = legacy_preview_corners(corners, (16805, 11429), (1200, 816))

        self.assertEqual(
            [[1165, 744], [1199, 815], [0, 0], [571, 356]], preview
        )


class CoordinateTransformTests(unittest.TestCase):
    def setUp(self):
        self.transform = DisplayTransform(
            scale=0.08,
            offset_x=100,
            offset_y=20,
            original_width=16805,
            original_height=11429,
        )

    def test_main_preview_round_trip_is_bounded_by_display_pixel_quantization(self):
        original = [16319, 10430]

        display = original_to_display(original, self.transform)
        restored = display_to_original(display, self.transform)

        maximum_error = math.ceil(0.5 / self.transform.scale)
        self.assertLessEqual(abs(restored[0] - original[0]), maximum_error)
        self.assertLessEqual(abs(restored[1] - original[1]), maximum_error)

    def test_original_to_display_uses_python_round_with_negative_offsets(self):
        transform = DisplayTransform(0.5, -3, -4, 5, 5)

        self.assertEqual((-2, -4), original_to_display([1, 1], transform))

    def test_transform_preserves_valid_original_pixel_boundaries(self):
        transform = DisplayTransform(1.0, -2, 3, 5, 4)

        self.assertEqual((-2, 3), original_to_display([0, 0], transform))
        self.assertEqual((2, 6), original_to_display([4, 3], transform))
        self.assertEqual([0, 0], display_to_original([-100, -100], transform))
        self.assertEqual([4, 3], display_to_original([100, 100], transform))

    def test_transform_rejects_non_finite_or_non_positive_scale(self):
        for scale in (0, -0.1, math.inf, -math.inf, math.nan):
            with self.subTest(scale=scale):
                with self.assertRaisesRegex(ValueError, "scale"):
                    DisplayTransform(scale, 0, 0, 10, 10)

    def test_transform_rejects_non_positive_original_dimensions(self):
        for width, height in ((0, 10), (-1, 10), (10, 0), (10, -1)):
            with self.subTest(width=width, height=height):
                with self.assertRaisesRegex(ValueError, "original"):
                    DisplayTransform(1, 0, 0, width, height)

    def test_transform_rejects_non_finite_offsets(self):
        for offset_x, offset_y in ((math.inf, 0), (0, math.nan)):
            with self.subTest(offset_x=offset_x, offset_y=offset_y):
                with self.assertRaisesRegex(ValueError, "offset"):
                    DisplayTransform(1, offset_x, offset_y, 10, 10)

    def test_coordinate_functions_require_exactly_two_finite_values(self):
        invalid_points = ([1], [1, 2, 3], [math.nan, 1], [1, math.inf])

        for function in (original_to_display, display_to_original):
            for point in invalid_points:
                with self.subTest(function=function.__name__, point=point):
                    with self.assertRaisesRegex(ValueError, "point"):
                        function(point, self.transform)


class CornerMovementTests(unittest.TestCase):
    def test_arrow_delta_moves_exactly_one_original_pixel(self):
        self.assertEqual([400, 317], move_corner([401, 317], -1, 0, (16805, 11429)))
        self.assertEqual([401, 318], move_corner([401, 317], 0, 1, (16805, 11429)))

    def test_movement_is_clamped_to_valid_original_coordinate(self):
        self.assertEqual([0, 11428], move_corner([0, 11428], -1, 1, (16805, 11429)))
        self.assertEqual([16804, 0], move_corner([16804, 0], 1, -1, (16805, 11429)))

    def test_movement_requires_integer_deltas_without_truncation(self):
        for dx, dy in ((0.5, 0), (1.0, 0), (0, "1"), (True, 0)):
            with self.subTest(dx=dx, dy=dy):
                with self.assertRaisesRegex(ValueError, "dx and dy"):
                    move_corner([4, 4], dx, dy, (10, 10))

    def test_movement_rejects_invalid_corner_or_image_size(self):
        cases = (
            ([1], (10, 10), "corner"),
            ([1, math.inf], (10, 10), "corner"),
            ([1, 2], (0, 10), "image_size"),
            ([1, 2], (10, -1), "image_size"),
        )

        for corner, image_size, message in cases:
            with self.subTest(corner=corner, image_size=image_size):
                with self.assertRaisesRegex(ValueError, message):
                    move_corner(corner, 0, 0, image_size)


class MagnifierTests(unittest.TestCase):
    def setUp(self):
        y, x = np.mgrid[0:200, 0:300]
        self.image = np.dstack((x % 256, y % 256, (x + y) % 256)).astype(
            np.uint8
        )

    def test_each_zoom_returns_unscaled_original_pixels(self):
        for zoom, expected_size in ((4, 100), (8, 50), (16, 25)):
            with self.subTest(zoom=zoom):
                source = render_magnifier_source(
                    self.image, [150, 100], zoom, viewport_size=400
                )

                self.assertEqual((expected_size, expected_size, 3), source.shape)

        np.testing.assert_array_equal(
            self.image[50:150, 100:200],
            render_magnifier_source(self.image, [150, 100], 4, 400),
        )

    def test_center_pixel_position_is_defined_for_even_and_odd_source_sizes(self):
        even_source = render_magnifier_source(self.image, [150, 100], 4, 400)
        odd_source = render_magnifier_source(self.image, [150, 100], 16, 400)

        np.testing.assert_array_equal(self.image[100, 150], even_source[50, 50])
        np.testing.assert_array_equal(self.image[100, 150], odd_source[12, 12])
        np.testing.assert_array_equal(self.image[99, 149], even_source[49, 49])
        np.testing.assert_array_equal(self.image[99, 149], odd_source[11, 11])

    def test_all_corners_are_black_padded_with_center_pixel_preserved(self):
        image = (np.arange(5 * 6 * 2).reshape(5, 6, 2) + 1).astype(np.uint16)
        cases = (
            ((0, 0), (slice(0, 2), slice(0, 2)), (slice(2, 4), slice(2, 4))),
            ((5, 0), (slice(0, 2), slice(3, 6)), (slice(2, 4), slice(0, 3))),
            ((0, 4), (slice(2, 5), slice(0, 2)), (slice(0, 3), slice(2, 4))),
            ((5, 4), (slice(2, 5), slice(3, 6)), (slice(0, 3), slice(0, 3))),
        )

        for center, image_slices, source_slices in cases:
            with self.subTest(center=center):
                source = render_magnifier_source(image, center, 4, viewport_size=16)
                expected = np.zeros((4, 4, 2), dtype=np.uint16)
                expected[source_slices] = image[image_slices]

                self.assertEqual(image.dtype, source.dtype)
                self.assertEqual(image.shape[2:], source.shape[2:])
                np.testing.assert_array_equal(expected, source)
                np.testing.assert_array_equal(
                    image[center[1], center[0]], source[2, 2]
                )

    def test_grayscale_padding_preserves_dtype_and_input_is_not_aliased(self):
        image = np.arange(20, dtype=np.int16).reshape(4, 5)
        original = image.copy()

        source = render_magnifier_source(image, [2, 2], 4, viewport_size=16)
        source[:] = -1

        self.assertEqual(np.int16, source.dtype)
        self.assertEqual((4, 4), source.shape)
        np.testing.assert_array_equal(original, image)

    def test_rejects_empty_or_wrong_dimension_images(self):
        invalid_images = (
            np.empty((0, 3), dtype=np.uint8),
            np.empty((3, 0, 3), dtype=np.uint8),
            np.empty((3, 3, 0), dtype=np.uint8),
            np.empty((3,), dtype=np.uint8),
            np.empty((1, 2, 3, 4), dtype=np.uint8),
            [[1, 2], [3, 4]],
        )

        for image in invalid_images:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaisesRegex(ValueError, "image"):
                    render_magnifier_source(image, [0, 0], 4, 400)

    def test_rejects_non_integer_or_out_of_bounds_center(self):
        invalid_centers = (
            [1],
            [1, 2, 3],
            [1.0, 2],
            [True, 2],
            [-1, 2],
            [300, 2],
            [2, -1],
            [2, 200],
        )

        for center in invalid_centers:
            with self.subTest(center=center):
                with self.assertRaisesRegex(ValueError, "center"):
                    render_magnifier_source(self.image, center, 4, 400)

    def test_rejects_invalid_zoom_or_viewport_size(self):
        for zoom in (True, 4.0, 1, 32):
            with self.subTest(zoom=zoom):
                with self.assertRaisesRegex(ValueError, "zoom"):
                    render_magnifier_source(self.image, [1, 1], zoom, 400)

        for viewport_size in (True, 400.0, 0, -4, 402):
            with self.subTest(viewport_size=viewport_size):
                with self.assertRaisesRegex(ValueError, "viewport_size"):
                    render_magnifier_source(
                        self.image, [1, 1], zoom=4, viewport_size=viewport_size
                    )

    def test_drag_is_inverse_and_quantized_to_original_pixels(self):
        self.assertEqual((-1, 2), magnifier_drag_delta(8, -16, zoom=8))
        self.assertEqual((1, -2), magnifier_drag_delta(-8, 16, zoom=8))

    def test_drag_uses_python_round_ties_to_even_and_ignores_small_motion(self):
        cases = (
            ((3, -3), (0, 0)),
            ((4, -4), (0, 0)),
            ((12, -12), (-2, 2)),
            ((-12, 12), (2, -2)),
        )

        for screen_delta, expected in cases:
            with self.subTest(screen_delta=screen_delta):
                self.assertEqual(
                    expected,
                    magnifier_drag_delta(*screen_delta, zoom=8),
                )

    def test_drag_rejects_non_integer_delta_bool_or_invalid_zoom(self):
        for screen_dx, screen_dy, zoom in (
            (True, 0, 4),
            (0, False, 4),
            (1.0, 0, 4),
            (0, 1.0, 4),
            (0, 0, True),
            (0, 0, 4.0),
            (0, 0, 3),
        ):
            with self.subTest(
                screen_dx=screen_dx, screen_dy=screen_dy, zoom=zoom
            ):
                with self.assertRaises(ValueError):
                    magnifier_drag_delta(screen_dx, screen_dy, zoom)


if __name__ == "__main__":
    unittest.main()
