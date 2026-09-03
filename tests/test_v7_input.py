import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from photocut.algorithms.v7.input import (
    InputDecodeError,
    PairedInputViews,
    decode_bytes,
    float_corners_to_display,
)


def _jpeg(mode="RGB", orientation=None):
    image = Image.new(mode, (4, 3))
    if mode == "L":
        image.putdata(range(12))
    else:
        image.putdata([(x * 20, y * 30, 80) for y in range(3) for x in range(4)])
    out = io.BytesIO()
    image.save(out, format="JPEG", exif=(b"" if orientation is None else _exif(orientation)))
    return out.getvalue()


def _exif(orientation):
    image = Image.new("RGB", (1, 1))
    exif = image.getexif()
    exif[274] = orientation
    return exif.tobytes()


def test_decode_hashes_bytes_and_normalizes_rgb_to_readonly_bgr():
    data = _jpeg(orientation=1)
    loaded = decode_bytes(data)
    assert loaded.source_sha256 == hashlib.sha256(data).hexdigest()
    assert loaded.original_dtype == "uint8"
    assert loaded.original_channels == 3
    assert loaded.exif_orientation == 1
    assert loaded.normalized_bgr.shape == (3, 4, 3)
    assert loaded.normalized_bgr.flags.writeable is False
    with pytest.raises(ValueError):
        loaded.normalized_bgr[0, 0, 0] = 1


@pytest.mark.parametrize("orientation", range(1, 9))
def test_all_exif_orientations_have_expected_shape_and_inverse(orientation):
    loaded = decode_bytes(_jpeg(orientation=orientation))
    # Rotations 5-8 swap dimensions; flips preserve dimensions.
    expected = (4, 3) if orientation in (5, 6, 7, 8) else (3, 4)
    assert loaded.normalized_bgr.shape[:2] == expected
    points = ((0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0))
    mapped = loaded.map_original_to_normalized(points)
    round_trip = loaded.map_normalized_to_original(mapped)
    np.testing.assert_allclose(round_trip, points, atol=1e-6)


@pytest.mark.parametrize("orientation", (5, 6, 7, 8))
def test_tiff_exif_rotations_use_stored_dimensions(orientation):
    image = Image.new("RGB", (4, 3), (20, 40, 60))
    exif = image.getexif(); exif[274] = orientation
    out = io.BytesIO(); image.save(out, format="TIFF", exif=exif.tobytes())
    loaded = decode_bytes(out.getvalue())
    assert loaded.original_size == (4, 3)
    assert loaded.normalized_size == (3, 4)
    points = ((0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0))
    np.testing.assert_allclose(loaded.map_normalized_to_original(
        loaded.map_original_to_normalized(points)), points, atol=1e-6)


def test_grayscale_rgba_and_lossless_uint16_conversion():
    gray = decode_bytes(_jpeg("L"))
    assert gray.original_channels == 1 and gray.normalized_bgr.shape[-1] == 3

    rgba = Image.new("RGBA", (2, 2), (1, 2, 3, 4))
    out = io.BytesIO(); rgba.save(out, format="PNG")
    rgba_loaded = decode_bytes(out.getvalue())
    assert rgba_loaded.original_channels == 4
    assert tuple(rgba_loaded.normalized_bgr[0, 0]) == (3, 2, 1)

    sixteen = Image.new("I;16", (2, 2), 200)
    out = io.BytesIO(); sixteen.save(out, format="TIFF")
    converted = decode_bytes(out.getvalue())
    assert converted.original_dtype == "uint16"
    assert converted.normalized_bgr.dtype == np.uint8

    big_endian = Image.fromarray(np.array([[200, 10], [20, 30]], dtype=">u2"), mode="I;16B")
    out = io.BytesIO(); big_endian.save(out, format="TIFF")
    converted_be = decode_bytes(out.getvalue())
    assert converted_be.normalized_bgr.dtype == np.uint8

    too_large = Image.new("I;16", (2, 2), 1000)
    out = io.BytesIO(); too_large.save(out, format="TIFF")
    with pytest.raises(InputDecodeError, match="16-bit"):
        decode_bytes(out.getvalue())


def test_invalid_corrupt_and_pixel_budget_fail():
    with pytest.raises(InputDecodeError):
        decode_bytes(b"not an image")
    with pytest.raises(InputDecodeError, match="pixel"):
        decode_bytes(_jpeg(), max_pixel_budget=2)


def test_display_round_half_up_then_clamp():
    got = float_corners_to_display(((-1.2, 0.49), (2.5, 3.5), (9.9, 8.9), (1.49, 1.5)), (4, 4))
    assert got == ((0, 0), (3, 3), (3, 3), (1, 2))


def test_paired_views_use_actual_legacy_loader(tmp_path, monkeypatch):
    path = tmp_path / "photo.jpg"
    path.write_bytes(_jpeg(orientation=6))
    sentinel = np.zeros((3, 4, 3), dtype=np.uint8)
    monkeypatch.setattr("photocut.core.load_image", lambda p: sentinel)
    pair = PairedInputViews.from_path(path)
    np.testing.assert_array_equal(pair.legacy_array, sentinel)
    assert pair.legacy_array.flags.writeable is False
    assert pair.v7_array.shape[:2] == (4, 3)
    mapped = pair.map_legacy_to_normalized(((0, 0), (3, 2)))
    assert len(mapped) == 2


def test_paired_views_real_loader_orientation_shape_and_transform(tmp_path):
    path = tmp_path / "oriented.jpg"
    path.write_bytes(_jpeg(orientation=6))
    pair = PairedInputViews.from_path(path)
    assert tuple(pair.legacy_array.shape[:2]) in {
        tuple(pair.loaded.normalized_bgr.shape[:2]),
        (pair.loaded.original_size[1], pair.loaded.original_size[0]),
    }
