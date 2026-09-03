import cv2
import numpy as np
import pytest

from photocut.algorithms.v7.model_inference import letterbox_rgb_nchw
from photocut.algorithms.v8.mask_postprocess import MaskPostprocessError, decode_photo_mask
from photocut.algorithms.v8.training_data import rasterize_photo_mask


def _logits(mask):
    foreground = np.where(mask == 1, 8.0, -8.0).astype(np.float32)
    return np.stack((-foreground, foreground), axis=0)[None]


def test_mask_postprocess_keeps_one_main_component_and_recovers_perspective_quad():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    _, transform = letterbox_rgb_nchw(image, (128, 128))
    original = ((10, 8), (91, 14), (84, 72), (15, 68))
    model_quad = transform.to_model(original)
    mask = rasterize_photo_mask((128, 128), model_quad)
    cv2.rectangle(mask, (2, 16), (7, 21), 1, -1)

    result = decode_photo_mask(
        {"mask_logits": _logits(mask)},
        transform,
        image_size=(100, 80),
    )

    assert result is not None
    np.testing.assert_allclose(result.corners, original, atol=3.0)
    assert result.confidence > 0.8
    assert result.evidence["component_count"] == 2
    assert result.evidence["main_component_ratio"] > 0.95
    assert result.evidence["polygon_mask_iou"] > 0.9


def test_mask_postprocess_returns_no_candidate_for_empty_foreground():
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    _, transform = letterbox_rgb_nchw(image, (64, 64))
    logits = np.zeros((1, 2, 64, 64), dtype=np.float32)
    logits[:, 0] = 8
    logits[:, 1] = -8

    assert decode_photo_mask({"mask_logits": logits}, transform, image_size=(60, 40)) is None


@pytest.mark.parametrize(
    "outputs",
    [
        {},
        {"mask_logits": np.zeros((1, 1, 32, 32), np.float32)},
        {"mask_logits": np.full((1, 2, 32, 32), np.nan, np.float32)},
    ],
)
def test_mask_postprocess_rejects_malformed_logits(outputs):
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    _, transform = letterbox_rgb_nchw(image, (32, 32))
    with pytest.raises(MaskPostprocessError):
        decode_photo_mask(outputs, transform, image_size=(30, 20))
