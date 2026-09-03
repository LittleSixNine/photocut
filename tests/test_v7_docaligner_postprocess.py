import unittest

import numpy as np

from photocut.algorithms.v7.model_inference import LetterboxTransform
from photocut.algorithms.v7.model_postprocess import ModelOutputError, decode_docaligner_outputs


def _heatmaps(corners=((8, 8), (55, 8), (55, 55), (8, 55))):
    output = np.zeros((1, 4, 64, 64), dtype=np.float32)
    for channel, (x, y) in enumerate(corners):
        output[0, channel, y:y + 2, x:x + 2] = 0.9
    return {"heatmap": output}


class DocAlignerPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.transform = LetterboxTransform((256, 256), (256, 256), 1.0, 0, 0)

    def test_decodes_largest_supported_component_and_evidence(self):
        outputs = _heatmaps()
        # A smaller distracting response must not replace the main component.
        outputs["heatmap"][0, 0, 40, 40] = 1.0
        proposals = decode_docaligner_outputs(outputs, self.transform, (256, 256))
        self.assertEqual(1, len(proposals))
        proposal = proposals[0]
        self.assertEqual("heatmaps", proposal.head)
        self.assertEqual(4, len(proposal.corners))
        self.assertEqual(4, len(proposal.evidence["corner_support_pixels"]))
        self.assertGreater(proposal.evidence["min_peak_sigma"], 5.0)

    def test_missing_or_diffuse_corner_returns_no_candidate(self):
        outputs = _heatmaps()
        outputs["heatmap"][0, 2].fill(0.0)
        self.assertEqual((), decode_docaligner_outputs(
            outputs, self.transform, (256, 256)
        ))

    def test_maps_letterboxed_portrait_back_to_original_coordinates(self):
        transform = LetterboxTransform((100, 200), (256, 256), 1.28, 64, 0)
        outputs = _heatmaps(((17, 8), (46, 8), (46, 55), (17, 55)))
        proposal = decode_docaligner_outputs(outputs, transform, (100, 200))[0]
        self.assertTrue(all(0 <= x <= 99 and 0 <= y <= 199 for x, y in proposal.corners))

    def test_rejects_malformed_output(self):
        for value in (
            np.zeros((4, 64, 64), np.float32),
            np.zeros((1, 3, 64, 64), np.float32),
            np.full((1, 4, 64, 64), np.nan, np.float32),
        ):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ModelOutputError):
                    decode_docaligner_outputs({"heatmap": value}, self.transform, (256, 256))


if __name__ == "__main__":
    unittest.main()
