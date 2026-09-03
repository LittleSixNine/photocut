"""Deterministic preprocessing and lazy optional ONNX inference for V7."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from .model_manifest import ModelManifest


class ModelInferenceError(RuntimeError):
    pass


class InferenceBackend(Protocol):
    def run(self, tensor: np.ndarray) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class LetterboxTransform:
    original_size: tuple[int, int]
    model_size: tuple[int, int]
    scale: float
    pad_x: int
    pad_y: int

    def to_model(self, points: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return tuple((float(x) * self.scale + self.pad_x,
                      float(y) * self.scale + self.pad_y) for x, y in points)

    def to_original(self, points: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
        return tuple(((float(x) - self.pad_x) / self.scale,
                      (float(y) - self.pad_y) / self.scale) for x, y in points)


def letterbox_rgb_nchw(image: np.ndarray, model_size: Sequence[int]) -> tuple[np.ndarray, LetterboxTransform]:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or not image.size or
            image.shape[0] <= 0 or image.shape[1] <= 0):
        raise ModelInferenceError("model input must be a non-empty uint8 BGR image")
    if (not isinstance(model_size, (tuple, list)) or len(model_size) != 2 or
            any(type(item) is not int or item <= 0 for item in model_size)):
        raise ModelInferenceError("model size must contain two positive integers")
    model_w, model_h = int(model_size[0]), int(model_size[1])
    height, width = image.shape[:2]
    scale = min(model_w / float(width), model_h / float(height))
    resized_w = max(1, min(model_w, int(round(width * scale))))
    resized_h = max(1, min(model_h, int(round(height * scale))))
    pad_x = (model_w - resized_w) // 2
    pad_y = (model_h - resized_h) // 2
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((model_h, model_w, 3), 128, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    rgb = canvas[:, :, ::-1]
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= np.float32(255.0)
    transform = LetterboxTransform((width, height), (model_w, model_h), scale, pad_x, pad_y)
    return tensor, transform


class OnnxRuntimeBackend:
    """A minimal CPU-only ONNX Runtime wrapper imported only when constructed."""

    def __init__(self, verified_model: Path, manifest: ModelManifest):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelInferenceError("onnxruntime is not installed") from exc
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        execution_mode = getattr(getattr(ort, "ExecutionMode", None), "ORT_SEQUENTIAL", None)
        if execution_mode is not None:
            options.execution_mode = execution_mode
        try:
            session = ort.InferenceSession(
                str(verified_model), sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise ModelInferenceError(f"cannot open ONNX model: {exc}") from exc
        inputs = tuple(item.name for item in session.get_inputs())
        outputs = tuple(item.name for item in session.get_outputs())
        if inputs != (manifest.input_name,):
            raise ModelInferenceError(f"model input signature mismatch: {inputs}")
        if len(outputs) != len(manifest.output_names) or set(outputs) != set(manifest.output_names):
            raise ModelInferenceError(f"model output signature mismatch: {outputs}")
        self._session = session
        self._manifest = manifest

    def run(self, tensor: np.ndarray) -> Mapping[str, np.ndarray]:
        expected_w, expected_h = self._manifest.input_size
        if (not isinstance(tensor, np.ndarray) or tensor.dtype != np.float32 or
                tensor.shape != (1, 3, expected_h, expected_w) or
                not tensor.flags.c_contiguous or not np.isfinite(tensor).all()):
            raise ModelInferenceError("inference tensor does not match manifest input")
        try:
            values = self._session.run(
                list(self._manifest.output_names),
                {self._manifest.input_name: tensor},
            )
        except Exception as exc:
            raise ModelInferenceError(f"ONNX inference failed: {exc}") from exc
        if len(values) != len(self._manifest.output_names):
            raise ModelInferenceError("ONNX output count mismatch")
        output: dict[str, np.ndarray] = {}
        for name, value in zip(self._manifest.output_names, values):
            array = np.array(value, copy=True)
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ModelInferenceError(f"ONNX output {name!r} is not finite numeric data")
            array.setflags(write=False)
            output[name] = array
        return output
