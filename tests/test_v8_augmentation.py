import json

import cv2
import numpy as np
import pytest

from photocut.algorithms.v8.training_data import (
    AugmentationConfig,
    build_augmentation_lineage,
    generate_synthetic_sample,
    rasterize_photo_mask,
)


def _fixture():
    image = np.full((72, 96, 3), (238, 241, 245), dtype=np.uint8)
    corners = ((14, 10), (82, 13), (78, 62), (17, 58))
    cv2.fillConvexPoly(
        image,
        np.rint(np.asarray(corners)).astype(np.int32),
        (40, 110, 190),
    )
    cv2.circle(image, (48, 36), 12, (210, 70, 50), -1)
    return image, corners


def test_synthetic_augmentation_is_byte_deterministic_for_one_seed():
    image, corners = _fixture()
    config = AugmentationConfig(output_size=(128, 96), synthetic_per_source=3)

    first = generate_synthetic_sample(
        image,
        corners,
        seed=1234,
        source_image_id="sha256:source",
        config=config,
    )
    second = generate_synthetic_sample(
        image,
        corners,
        seed=1234,
        source_image_id="sha256:source",
        config=config,
    )
    different = generate_synthetic_sample(
        image,
        corners,
        seed=1235,
        source_image_id="sha256:source",
        config=config,
    )

    assert first.lineage == second.lineage
    assert first.image_bgr.tobytes() == second.image_bgr.tobytes()
    assert first.mask.tobytes() == second.mask.tobytes()
    assert first.image_bgr.tobytes() != different.image_bgr.tobytes()


def test_synthetic_mask_stays_binary_and_matches_recorded_quad():
    image, corners = _fixture()
    config = AugmentationConfig(output_size=(128, 96), synthetic_per_source=1)

    result = generate_synthetic_sample(
        image,
        corners,
        seed=7,
        source_image_id="sha256:source",
        config=config,
    )

    expected = rasterize_photo_mask(config.output_size, result.corners)
    intersection = np.logical_and(result.mask, expected).sum()
    union = np.logical_or(result.mask, expected).sum()
    assert result.image_bgr.shape == (96, 128, 3)
    assert result.image_bgr.dtype == np.uint8
    assert result.mask.shape == (96, 128)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)) <= {0, 1}
    assert intersection / union > 0.985
    assert result.lineage["source_image_id"] == "sha256:source"
    assert result.lineage["recipe"] == "scanner_white_composite_v1"
    assert result.lineage["config_hash"].startswith("sha256:")
    json.dumps(result.lineage, allow_nan=False)


@pytest.mark.parametrize(
    "changes",
    [
        {"output_size": (1, 96)},
        {"synthetic_per_source": -1},
        {"max_rotation_deg": 90},
        {"max_perspective": 0.5},
    ],
)
def test_augmentation_config_rejects_unbounded_values(changes):
    with pytest.raises((TypeError, ValueError)):
        AugmentationConfig(**changes)


def test_lineage_uses_only_fit_as_synthetic_source_and_keeps_calibration_real():
    manifest = {
        "schema_version": 1,
        "manifest_id": "sha256:" + "1" * 64,
        "population_id": "sha256:" + "2" * 64,
        "samples": [
            {"image_id": "fit-a", "origin_group_id": "g1", "training_subset": "fit"},
            {"image_id": "fit-b", "origin_group_id": "g2", "training_subset": "fit"},
            {"image_id": "cal-a", "origin_group_id": "g3", "training_subset": "calibration"},
        ],
        "frozen_accessed": False,
    }
    config = AugmentationConfig(output_size=(128, 96), synthetic_per_source=2)

    first = build_augmentation_lineage(manifest, config=config, seed=20260831)
    second = build_augmentation_lineage(manifest, config=config, seed=20260831)

    assert first == second
    fit_synthetic = [row for row in first if row["kind"] == "synthetic"]
    real = [row for row in first if row["kind"] == "real"]
    assert len(fit_synthetic) == 4
    assert {row["source_image_id"] for row in fit_synthetic} == {"fit-a", "fit-b"}
    assert {row["source_image_id"] for row in real} == {"fit-a", "fit-b", "cal-a"}
    assert len({row["sample_id"] for row in first}) == len(first)

    bad = dict(manifest, frozen_accessed=True)
    with pytest.raises(PermissionError, match="frozen"):
        build_augmentation_lineage(bad, config=config, seed=20260831)
