import numpy as np

from photocut.algorithms.v8.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_mask_input,
    preprocess_mask_target,
)


def test_mask_preprocess_reuses_letterbox_and_imagenet_normalization_exactly():
    image = np.asarray([[[0, 64, 255], [255, 128, 0]]], dtype=np.uint8)  # BGR

    result = preprocess_mask_input(image, (2, 2))

    expected_rgb = np.asarray(
        [
            [[255, 64, 0], [0, 128, 255]],
            [[128, 128, 128], [128, 128, 128]],
        ],
        dtype=np.float32,
    )
    expected = expected_rgb.transpose(2, 0, 1)[None] / np.float32(255.0)
    expected = (expected - np.asarray(IMAGENET_MEAN, np.float32)[None, :, None, None])
    expected /= np.asarray(IMAGENET_STD, np.float32)[None, :, None, None]

    np.testing.assert_array_equal(result.tensor, expected)
    assert result.tensor.shape == (1, 3, 2, 2)
    assert result.tensor.dtype == np.float32
    assert result.tensor.flags.c_contiguous
    assert result.transform.original_size == (2, 1)


def test_mask_target_uses_same_transform_and_nearest_binary_values():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    mask = np.asarray([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint8)
    prepared = preprocess_mask_input(image, (8, 8))

    target = preprocess_mask_target(mask, prepared.transform)

    assert target.shape == (8, 8)
    assert target.dtype == np.uint8
    assert set(np.unique(target)) == {0, 1}
    assert np.all(target[:2] == 0)
    assert np.all(target[6:] == 0)
    assert target[3, 3] == 1
