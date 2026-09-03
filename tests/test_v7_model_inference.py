import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photocut.algorithms.v7.model_inference import (
    ModelInferenceError,
    OnnxRuntimeBackend,
    letterbox_rgb_nchw,
)
from photocut.algorithms.v7.model_manifest import ModelManifest


def _manifest(output_names=("heatmap",)):
    return ModelManifest(
        schema_version=1,
        model_id="test-model",
        adapter="docaligner_heatmap",
        source_repository="https://example.test/repo",
        source_revision="revision",
        license="Apache-2.0",
        notice_path="LICENSE",
        model_filename="model.onnx",
        model_size=1,
        model_sha256="sha256:" + "0" * 64,
        input_name="img",
        input_size=(256, 256),
        output_names=tuple(output_names),
        redistributable=False,
    )


class ModelInferenceTests(unittest.TestCase):
    def test_letterbox_uses_mid_gray_and_round_trips_points(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        tensor, transform = letterbox_rgb_nchw(image, (256, 256))
        self.assertEqual((1, 3, 256, 256), tensor.shape)
        self.assertEqual(np.float32, tensor.dtype)
        self.assertTrue(tensor.flags.c_contiguous)
        # The 100x200 source occupies 128x256; top and bottom are padding.
        self.assertTrue(np.allclose(tensor[:, :, :64, :], 128.0 / 255.0))
        source = ((0.0, 0.0), (199.0, 99.0), (73.25, 51.5))
        self.assertTrue(np.allclose(transform.to_original(transform.to_model(source)), source))

    def test_letterbox_converts_bgr_to_rgb(self):
        bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)
        tensor, _ = letterbox_rgb_nchw(bgr, (1, 1))
        self.assertTrue(np.allclose(tensor[0, :, 0, 0], [3 / 255, 2 / 255, 1 / 255]))

    def test_letterbox_rejects_malformed_images_and_sizes(self):
        for image, size in (
            (np.empty((0, 1, 3), np.uint8), (256, 256)),
            (np.zeros((2, 2), np.uint8), (256, 256)),
            (np.zeros((2, 2, 3), np.float32), (256, 256)),
            (np.zeros((2, 2, 3), np.uint8), (0, 256)),
        ):
            with self.subTest(shape=image.shape, size=size):
                with self.assertRaises(ModelInferenceError):
                    letterbox_rgb_nchw(image, size)

    def test_backend_is_lazy_and_returns_owned_named_outputs(self):
        calls = {}

        class Meta:
            def __init__(self, name):
                self.name = name

        class SessionOptions:
            pass

        class Session:
            def __init__(self, path, sess_options, providers):
                calls["path"] = path
                calls["options"] = sess_options
                calls["providers"] = providers

            def get_inputs(self):
                return [Meta("img")]

            def get_outputs(self):
                return [Meta("heatmap")]

            def run(self, names, feed):
                calls["run"] = (names, feed)
                return [np.ones((1, 4, 64, 64), dtype=np.float32)]

        fake = types.SimpleNamespace(
            SessionOptions=SessionOptions,
            InferenceSession=Session,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        )
        with patch.dict(sys.modules, {"onnxruntime": fake}):
            backend = OnnxRuntimeBackend(Path("/verified/model.onnx"), _manifest())
            output = backend.run(np.zeros((1, 3, 256, 256), np.float32))
        self.assertEqual(["CPUExecutionProvider"], calls["providers"])
        self.assertEqual(["heatmap"], calls["run"][0])
        self.assertEqual({"img"}, set(calls["run"][1]))
        self.assertEqual({"heatmap"}, set(output))
        self.assertTrue(output["heatmap"].flags.owndata)
        self.assertFalse(output["heatmap"].flags.writeable)

    def test_backend_rejects_runtime_signature_and_output_mismatches(self):
        class Meta:
            def __init__(self, name):
                self.name = name

        class SessionOptions:
            pass

        class Session:
            input_name = "wrong"
            output_names = ("heatmap",)

            def __init__(self, *args, **kwargs):
                pass

            def get_inputs(self):
                return [Meta(self.input_name)]

            def get_outputs(self):
                return [Meta(name) for name in self.output_names]

        fake = types.SimpleNamespace(
            SessionOptions=SessionOptions,
            InferenceSession=Session,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        )
        with patch.dict(sys.modules, {"onnxruntime": fake}):
            with self.assertRaisesRegex(ModelInferenceError, "input"):
                OnnxRuntimeBackend(Path("/verified/model.onnx"), _manifest())
            Session.input_name = "img"
            Session.output_names = ("wrong",)
            with self.assertRaisesRegex(ModelInferenceError, "output"):
                OnnxRuntimeBackend(Path("/verified/model.onnx"), _manifest())

    def test_backend_accepts_runtime_output_order_because_outputs_are_read_by_name(self):
        class Meta:
            def __init__(self, name):
                self.name = name

        class SessionOptions:
            pass

        class Session:
            def __init__(self, *args, **kwargs):
                pass

            def get_inputs(self):
                return [Meta("img")]

            def get_outputs(self):
                return [Meta("mask_logits"), Meta("corner_heatmaps")]

        fake = types.SimpleNamespace(
            SessionOptions=SessionOptions,
            InferenceSession=Session,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        )
        with patch.dict(sys.modules, {"onnxruntime": fake}):
            OnnxRuntimeBackend(Path("/verified/model.onnx"), _manifest(
                ("corner_heatmaps", "mask_logits")
            ))


if __name__ == "__main__":
    unittest.main()
