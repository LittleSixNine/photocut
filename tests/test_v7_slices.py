import unittest

from photocut.algorithms.v7.slices import derive_slices, slice_membership


class SliceTests(unittest.TestCase):
    def test_labels_overlap_and_origin_group_is_independent_unit(self):
        result = derive_slices(
            audited_labels={"slices": ["weak_border", "complex_texture"], "rotation_degrees": 12},
            image_id="img-1", origin_group_id="group-1",
        )
        self.assertEqual("group-1", result["independent_unit"])
        self.assertEqual({"weak_border", "complex_texture", "rotated"}, set(result["slices"]))

    def test_true_perspective_and_touch_border_are_derived_from_geometry(self):
        result = derive_slices(
            audited_labels={}, truth=[(0, 0), (100, 5), (90, 90), (2, 100)], image_size=(100, 100)
        )
        self.assertIn("perspective", result["slices"])
        self.assertIn("touches_border", result["slices"])

    def test_manifest_sample_projection_preserves_ids(self):
        result = slice_membership({
            "image_id": "img", "origin_group_id": "origin", "audited_labels": {"slices": ["outer_frame"]},
            "photo_truth": [[10, 10], [90, 10], [90, 90], [10, 90]],
        })
        self.assertEqual("img", result["image_id"])
        self.assertIn("outer_frame", result["slices"])


if __name__ == "__main__":
    unittest.main()
