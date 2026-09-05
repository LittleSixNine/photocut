"""V8.4 release contracts using synthetic images only."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from photocut.algorithms.v8_4.runtime import V84Runtime
from photocut.algorithms.v8.preprocess import preprocess_mask_input


class V84RuntimeTests(unittest.TestCase):
    def loaded(self, constant=False):
        image = np.full((128, 128, 3), 128, dtype=np.uint8)
        if not constant:
            image[32:96, 32:96] = (40, 70, 110)
        return SimpleNamespace(normalized_bgr=image, normalized_size=(128, 128))

    def runtime(self, points=None):
        runtime = V84Runtime.__new__(V84Runtime)
        if points is None:
            points = [[-.8, -.8], [.8, -.8], [.8, .8], [-.8, .8]]
        runtime.session = Mock()
        runtime.session.run.return_value = [np.asarray([points], dtype=np.float32)]
        return runtime

    def test_constant_image_is_rejected_before_inference(self):
        runtime = self.runtime()
        result = runtime.predict(self.loaded(constant=True))
        self.assertEqual(result['status'], 'no_boundary_evidence')
        self.assertIsNone(result['analysis_corners'])
        runtime.session.run.assert_not_called()

    def test_legal_candidate_is_analysis_pixels_and_requires_confirmation(self):
        runtime = self.runtime()
        result = runtime.predict(self.loaded())
        self.assertEqual(result['status'], 'candidate_requires_confirmation')
        self.assertTrue(np.allclose(result['analysis_corners'][0], [12.7375, 12.7375]))
        feed = runtime.session.run.call_args.args[1]['image']
        self.assertEqual(feed.shape, (1, 3, 1024, 1024))
        self.assertEqual(feed.dtype, np.float32)
        self.assertEqual(runtime.algorithm_version, '8.4')

    def test_letterbox_maps_landscape_and_portrait_to_analysis_pixels(self):
        for width, height in ((240, 120), (120, 240)):
            with self.subTest(size=(width, height)):
                image = np.full((height, width, 3), 128, dtype=np.uint8)
                image[20:80, 20:80] = 40
                loaded = SimpleNamespace(normalized_bgr=image, normalized_size=(width, height))
                truth = [[10, 10], [width - 11, 10], [width - 11, height - 11], [10, height - 11]]
                prepared = preprocess_mask_input(image, (1024, 1024))
                model_points = (np.asarray(prepared.transform.to_model(truth)) + .5) / 512 - 1
                result = self.runtime(model_points).predict(loaded)
                self.assertEqual(result['status'], 'candidate_requires_confirmation')
                self.assertTrue(np.allclose(result['analysis_corners'], truth, atol=1e-4))

    def test_out_of_frame_is_rejected_without_clipping(self):
        runtime = self.runtime([[-1.01, -.8], [.8, -.8], [.8, .8], [-1.01, .8]])
        result = runtime.predict(self.loaded())
        self.assertEqual(result['status'], 'invalid_geometry')
        self.assertIsNone(result['analysis_corners'])

    def test_self_crossing_quad_is_rejected(self):
        runtime = self.runtime([[-.8, -.8], [.8, .8], [.8, -.8], [-.8, .8]])
        self.assertEqual(runtime.predict(self.loaded())['status'], 'invalid_geometry')

    def test_nonfinite_and_wrong_shape_raise(self):
        runtime = self.runtime()
        for output in (np.zeros((4, 2)), np.full((1, 4, 2), np.nan)):
            with self.subTest(shape=output.shape):
                runtime.session.run.return_value = [output]
                with self.assertRaises(ValueError):
                    runtime.predict(self.loaded())

    def test_hash_mismatch_stops_before_loading_session(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            manifest = {'algorithm_version': '8.4', 'input_shape': [1, 3, 1024, 1024],
                        'output_shape': [1, 4, 2], 'coordinate_system': 'normalized_minus1_plus1_half_pixel',
                        'model_sha256': 'sha256:' + hashlib.sha256(b'expected').hexdigest()}
            (path / 'manifest.json').write_text(json.dumps(manifest))
            (path / 'model.onnx').write_bytes(b'tampered')
            with patch('photocut.algorithms.v8_4.runtime.ort.InferenceSession') as session:
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    V84Runtime(path)
                session.assert_not_called()


if __name__ == '__main__':
    unittest.main()
