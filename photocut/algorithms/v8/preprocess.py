"""Shared NumPy preprocessing for V8 mask training and ONNX inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from photocut.algorithms.v7.model_inference import LetterboxTransform, ModelInferenceError, letterbox_rgb_nchw


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class MaskPreprocessResult:
    tensor: np.ndarray
    transform: LetterboxTransform


def preprocess_mask_input(
    image_bgr: np.ndarray,
    model_size: Sequence[int] = (768, 768),
) -> MaskPreprocessResult:
    """Letterbox BGR input, convert to RGB, then apply ImageNet normalization."""
    tensor, transform = letterbox_rgb_nchw(image_bgr, model_size)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[None, :, None, None]
    normalized = np.ascontiguousarray((tensor - mean) / std, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ModelInferenceError("V8 mask preprocessing produced non-finite values")
    return MaskPreprocessResult(normalized, transform)


def preprocess_mask_target(mask: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    """Apply the exact image letterbox transform to a binary training mask."""
    value = np.asarray(mask)
    if (
        value.ndim != 2
        or value.dtype not in (np.uint8, np.bool_)
        or tuple((value.shape[1], value.shape[0])) != tuple(transform.original_size)
        or not set(np.unique(value)).issubset({0, 1})
    ):
        raise ValueError("mask must be binary and match transform.original_size")
    model_width, model_height = transform.model_size
    original_width, original_height = transform.original_size
    resized_width = max(1, min(model_width, int(round(original_width * transform.scale))))
    resized_height = max(1, min(model_height, int(round(original_height * transform.scale))))
    resized = cv2.resize(
        value.astype(np.uint8, copy=False),
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    output = np.zeros((model_height, model_width), dtype=np.uint8)
    output[
        transform.pad_y:transform.pad_y + resized_height,
        transform.pad_x:transform.pad_x + resized_width,
    ] = resized
    return output


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "MaskPreprocessResult",
    "preprocess_mask_input",
    "preprocess_mask_target",
]
