import unittest

import numpy as np

from photocut.algorithms.v7.scoring import score_components


def _candidate(sources):
    return {
        "candidate_id": "c1",
        "corners": ((10, 10), (90, 10), (90, 90), (10, 90)),
        "sources": tuple(sources),
        "providers": tuple(sources),
    }


class ModelAgreementTests(unittest.TestCase):
    def test_model_and_classical_provider_agreement_is_explicit_evidence(self):
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        agreed = score_components(_candidate(("docquadnet", "contour")), image, (100, 100))
        model_only = score_components(_candidate(("docquadnet",)), image, (100, 100))
        self.assertEqual(1.0, agreed["model_classical_agreement"])
        self.assertEqual(0.0, model_only["model_classical_agreement"])
        self.assertLessEqual(agreed["model_classical_agreement_weight"], 0.08)

    def test_agreement_never_masks_weak_edge_gate(self):
        image = np.full((100, 100, 3), 200, dtype=np.uint8)
        score = score_components(_candidate(("docquadnet", "contour")), image, (100, 100))
        self.assertLess(score["edge_score"], 0.35)
        self.assertLess(score["pre_score"], 0.35)


if __name__ == "__main__":
    unittest.main()
