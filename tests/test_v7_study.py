import unittest

from photocut.algorithms.v7.study import (
    assign_study_arms, blinded_view, build_study_event, assignment_hash,
)


class StudyTests(unittest.TestCase):
    def samples(self):
        return [
            {"image_id": "i1", "origin_group_id": "g1", "audited_labels": {"slices": ["rotated"]}},
            {"image_id": "i2", "origin_group_id": "g2", "audited_labels": {"slices": ["rotated"]}},
            {"image_id": "i3", "origin_group_id": "g3", "audited_labels": {"slices": ["weak_border"]}},
            {"image_id": "i4", "origin_group_id": "g4", "audited_labels": {"slices": ["weak_border"]}},
        ]

    def test_seed_reproducible_and_origin_group_stays_in_one_arm(self):
        first = assign_study_arms(self.samples(), seed=17)
        second = assign_study_arms(self.samples(), seed=17)
        self.assertEqual(first, second)
        arms = {item["origin_group_id"]: item["arm_pseudonym"] for item in first["assignments"]}
        self.assertEqual(4, len(arms))
        self.assertEqual(4, len({item["image_id"] for item in first["assignments"]}))

    def test_assignment_hash_changes_with_seed(self):
        self.assertNotEqual(assignment_hash(assign_study_arms(self.samples(), seed=1)), assignment_hash(assign_study_arms(self.samples(), seed=2)))

    def test_blinded_view_hides_algorithm_name_and_status_but_keeps_risks(self):
        view = blinded_view({
            "arm_pseudonym": "arm_a", "initial_corners": [[1, 1], [2, 1], [2, 2], [1, 2]],
            "risks": ["weak_edge"], "status": "v7_recommended", "detector": "v7",
        })
        self.assertEqual("arm_a", view["arm_pseudonym"])
        self.assertIn("risks", view)
        self.assertNotIn("detector", view)
        self.assertNotIn("status", view)

    def test_study_event_has_sealed_assignment_and_observational_label(self):
        event = build_study_event(
            assignment_id="assign-1", arm_pseudonym="arm_a", image_id="i1",
            initial_corners=[[1, 1], [2, 1], [2, 2], [1, 2]],
            operation="dragged", duration_ms=120, jitter_threshold_px=2.0,
            final_corners=[[2, 1], [2, 1], [2, 2], [1, 2]],
        )
        self.assertEqual(1, event["schema_version"])
        self.assertEqual("study", event["event_type"])
        self.assertEqual("observational", event["evidence_type"])
        self.assertTrue(event["event_id"])


if __name__ == "__main__":
    unittest.main()
