import unittest

from photocut.algorithms.v7.evaluation import (
    corner_errors, jitter_threshold, metric_dictionary, moved_corner_count,
    paired_binary_rate, cluster_bootstrap_delta,
)


class EvaluationTests(unittest.TestCase):
    truth = [[0, 0], [100, 0], [100, 100], [0, 100]]

    def test_corner_errors_and_jitter_threshold_are_deterministic(self):
        result = corner_errors([[3, 0], [100, 0], [100, 100], [0, 100]], self.truth, image_size=(100, 100))
        self.assertEqual(1, result["jittered_corner_count"])
        self.assertEqual(max(2.0, .0005 * (200 ** .5)), jitter_threshold((100, 100), self.truth))
        self.assertEqual(1, moved_corner_count(self.truth, [[3, 0], [100, 0], [100, 100], [0, 100]], image_size=(100, 100)))

    def test_metric_dictionary_maps_fallback_failure_and_top5(self):
        result = metric_dictionary({"image_id": "i", "truth": self.truth, "status": "v52_fallback", "top1_corners": self.truth, "top5_contains_truth": True})
        self.assertTrue(result["fallback"])
        self.assertFalse(result["failed"])
        self.assertTrue(result["top5_recall"])

    def test_paired_gates_report_effect_and_bootstrap_assumptions(self):
        result = paired_binary_rate([True, True, False], [False, True, False])
        self.assertEqual(3, result["n"])
        bootstrap = cluster_bootstrap_delta([1, 1, 1], [0, 0, 0], seed=7, repetitions=100)
        self.assertTrue(bootstrap["passed"])
        self.assertEqual(7, bootstrap["seed"])


if __name__ == "__main__":
    unittest.main()
