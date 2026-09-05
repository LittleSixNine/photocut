"""V8.4 single-network ONNX inference with exact geometry rejection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from photocut.algorithms.v7.geometry import validate_quad
from photocut.algorithms.v8.preprocess import preprocess_mask_input


class V84Runtime:
    algorithm_version = "8.4"

    def __init__(self, model_dir=None):
        directory = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parents[2] / "models" / "v8_4"
        self.manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("algorithm_version") != self.algorithm_version
                or self.manifest.get("input_shape") != [1, 3, 1024, 1024]
                or self.manifest.get("output_shape") != [1, 4, 2]
                or self.manifest.get("coordinate_system") != "normalized_minus1_plus1_half_pixel"):
            raise ValueError("V8.4 model manifest contract mismatch")
        model_bytes = (directory / "model.onnx").read_bytes()
        self.model_sha256 = "sha256:" + hashlib.sha256(model_bytes).hexdigest()
        if self.model_sha256 != self.manifest.get("model_sha256"):
            raise ValueError("V8.4 model hash mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_bytes, sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "image" or inputs[0].shape != [1, 3, 1024, 1024]
                or inputs[0].type != "tensor(float)" or len(outputs) != 1
                or outputs[0].name != "corners" or outputs[0].shape != [1, 4, 2]
                or outputs[0].type != "tensor(float)"):
            raise ValueError("V8.4 ONNX input/output contract mismatch")

    def predict(self, loaded_input):
        image = loaded_input.normalized_bgr
        if float(cv2.meanStdDev(image)[1].max()) == 0.0:
            return {"status": "no_boundary_evidence", "analysis_corners": None,
                    "reason": "spatially_constant_image"}
        prepared = preprocess_mask_input(image, (1024, 1024))
        output = np.asarray(self.session.run(["corners"], {"image": prepared.tensor})[0])
        if output.shape != (1, 4, 2) or not np.isfinite(output).all():
            raise ValueError("V8.4 model produced invalid output shape or nonfinite coordinates")
        points = (output[0] + 1) * 512 - 0.5
        corners = prepared.transform.to_original(points)
        try:
            validate_quad(corners, loaded_input.normalized_size)
        except ValueError as exc:
            return {"status": "invalid_geometry", "analysis_corners": None, "reason": str(exc)}
        return {"status": "candidate_requires_confirmation", "analysis_corners": [list(point) for point in corners],
                "reason": None}
