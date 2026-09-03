import time
import unittest

import numpy as np

from photocut.algorithms.v7.features import ImageFeatureContext
from photocut.algorithms.v7.model_inference import ModelInferenceError
from photocut.algorithms.v7.model_manifest import ModelManifest
from photocut.algorithms.v7.model_provider import ModelQuadProvider
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.types import ProviderStatus


def _manifest(adapter="docquadnet", outputs=("corner_heatmaps", "mask_logits")):
    return ModelManifest(1, "model-test", adapter, "https://example.test", "rev",
                         "Apache-2.0", "LICENSE", "model.ort", 1,
                         "sha256:" + "0" * 64, "input", (256, 256), tuple(outputs), False)


def _outputs():
    heatmaps = np.zeros((1, 4, 64, 64), dtype=np.float32)
    for i, (x, y) in enumerate(((8, 8), (55, 8), (55, 55), (8, 55))):
        heatmaps[0, i, y, x] = 10
    mask = np.full((1, 1, 64, 64), -10, dtype=np.float32)
    mask[0, 0, 8:56, 8:56] = 10
    return {"corner_heatmaps": heatmaps, "mask_logits": mask}


class FakeBackend:
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs or _outputs()
        self.error = error
        self.calls = 0

    def run(self, tensor):
        self.calls += 1
        if self.error:
            raise self.error
        return self.outputs


class ModelProviderTests(unittest.TestCase):
    def setUp(self):
        self.context = ImageFeatureContext(np.zeros((256, 256, 3), np.uint8))
        self.params = V7Parameters(provider_work_limits={"docquadnet": 100},
                                   provider_timeout_ms={"docquadnet": 250})

    def tearDown(self):
        self.context.close()

    def test_emits_stable_model_candidates_and_evidence(self):
        provider = ModelQuadProvider(FakeBackend(), _manifest())
        first = provider.provide(self.context, self.params)
        second = provider.provide(self.context, self.params)
        self.assertEqual(ProviderStatus.SUCCESS, first.status)
        self.assertEqual([c["candidate_id"] for c in first.candidates],
                         [c["candidate_id"] for c in second.candidates])
        self.assertTrue(first.candidates[0]["source"].startswith("model-test:"))
        self.assertIn("min_peak_sigma", first.candidates[0]["evidence"])

    def test_cancellation_and_expired_deadline_do_not_run_backend(self):
        class Token:
            is_cancelled = True
        backend = FakeBackend()
        provider = ModelQuadProvider(backend, _manifest())
        cancelled = provider.provide(self.context, self.params, cancellation_token=Token())
        expired = provider.provide(self.context, self.params, deadline=0)
        self.assertEqual(ProviderStatus.CANCELLED, cancelled.status)
        self.assertEqual(ProviderStatus.TIMEOUT, expired.status)
        self.assertEqual(0, backend.calls)

    def test_backend_error_isolated_and_partial_outputs_are_discarded(self):
        provider = ModelQuadProvider(FakeBackend(error=ModelInferenceError("bad output")), _manifest())
        result = provider.provide(self.context, self.params)
        self.assertEqual(ProviderStatus.ERROR, result.status)
        self.assertEqual((), result.candidates)
        self.assertEqual("ModelInferenceError", result.error_code)

    def test_docaligner_adapter_is_supported(self):
        heatmap = np.zeros((1, 4, 64, 64), np.float32)
        for channel, (x, y) in enumerate(((8, 8), (55, 8), (55, 55), (8, 55))):
            heatmap[0, channel, y:y + 2, x:x + 2] = 0.9
        provider = ModelQuadProvider(FakeBackend({"heatmap": heatmap}),
                                     _manifest("docaligner_heatmap", ("heatmap",)))
        result = provider.provide(self.context, self.params)
        self.assertEqual(ProviderStatus.SUCCESS, result.status)
        self.assertEqual(1, len(result.candidates))


if __name__ == "__main__":
    unittest.main()
