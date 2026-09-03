import unittest

import numpy as np

from photocut.algorithms.v7.model_inference import LetterboxTransform
from photocut.algorithms.v7.model_postprocess import ModelOutputError, decode_docquad_outputs


def _outputs(corners=((8, 8), (55, 8), (55, 55), (8, 55)), mask=True):
    heatmaps = np.zeros((1, 4, 64, 64), dtype=np.float32)
    for channel, (x, y) in enumerate(corners):
        heatmaps[0, channel, y, x] = 10.0
        if x + 1 < 64:
            heatmaps[0, channel, y, x + 1] = 4.0
    logits = np.full((1, 1, 64, 64), -10.0, dtype=np.float32)
    if mask:
        logits[0, 0, 8:56, 8:56] = 10.0
    return {"corner_heatmaps": heatmaps, "mask_logits": logits}


class DocQuadPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.transform = LetterboxTransform((256, 256), (256, 256), 1.0, 0, 0)

    def test_emits_corner_and_mask_proposals_with_agreement(self):
        proposals = decode_docquad_outputs(_outputs(), self.transform, (256, 256))
        self.assertEqual(2, len(proposals))
        self.assertEqual(("corners", "mask"), tuple(item.head for item in proposals))
        self.assertLess(proposals[0].evidence["corner_mask_distance"], 0.03)
        self.assertGreater(proposals[0].evidence["min_peak_sigma"], 5.0)
        for proposal in proposals:
            self.assertEqual(4, len(proposal.corners))
            self.assertTrue(all(0 <= x <= 255 and 0 <= y <= 255 for x, y in proposal.corners))

    def test_empty_mask_keeps_only_independent_corner_proposal(self):
        proposals = decode_docquad_outputs(_outputs(mask=False), self.transform, (256, 256))
        self.assertEqual(("corners",), tuple(item.head for item in proposals))
        self.assertFalse(proposals[0].evidence["mask_available"])

    def test_diffuse_heatmaps_are_reported_as_uncertain(self):
        outputs = _outputs()
        outputs["corner_heatmaps"].fill(1.0)
        proposals = decode_docquad_outputs(outputs, self.transform, (256, 256))
        # The corner proposal is geometrically degenerate and rejected; the mask
        # proposal remains but carries the uncertainty rather than hiding it.
        self.assertEqual(("mask",), tuple(item.head for item in proposals))
        self.assertTrue(proposals[0].evidence["diffuse_heatmaps"])

    def test_five_by_five_refinement_moves_peak_subpixel(self):
        outputs = _outputs(mask=False)
        heatmap = outputs["corner_heatmaps"][0, 0]
        heatmap[8, 9] = 9.0
        proposal = decode_docquad_outputs(outputs, self.transform, (256, 256))[0]
        # Argmax center would be x=34; a strong right neighbor moves it right.
        self.assertGreater(proposal.corners[0][0], 34.0)

    def test_rejects_wrong_shapes_nonfinite_and_illegal_geometry(self):
        bad = [
            {"corner_heatmaps": np.zeros((1, 3, 64, 64), np.float32),
             "mask_logits": np.zeros((1, 1, 64, 64), np.float32)},
            {"corner_heatmaps": np.full((1, 4, 64, 64), np.nan, np.float32),
             "mask_logits": np.zeros((1, 1, 64, 64), np.float32)},
        ]
        for output in bad:
            with self.subTest(shape=output["corner_heatmaps"].shape):
                with self.assertRaises(ModelOutputError):
                    decode_docquad_outputs(output, self.transform, (256, 256))

        # Four coincident heatmap peaks plus no mask produce no legal proposal.
        self.assertEqual((), decode_docquad_outputs(
            _outputs(corners=((5, 5),) * 4, mask=False), self.transform, (256, 256)
        ))


if __name__ == "__main__":
    unittest.main()
