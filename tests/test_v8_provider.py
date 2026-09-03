import numpy as np
from pathlib import Path
from unittest.mock import patch

from photocut.algorithms.v7.features import ImageFeatureContext
from photocut.algorithms.v7.model_inference import letterbox_rgb_nchw
from photocut.algorithms.v7.model_manifest import ModelManifest
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.types import ProviderStatus
from photocut.algorithms.v8.preprocess import preprocess_mask_input
from photocut.algorithms.v8.provider import LazyOnnxRuntimeBackend, PhotoMaskProvider
from photocut.algorithms.v8.training_data import rasterize_photo_mask


def _manifest(size=(128, 128)):
    return ModelManifest(
        schema_version=1,
        model_id="v8-test-mask",
        adapter="photo_mask_v8",
        source_repository="local:test",
        source_revision="revision",
        license="research-only",
        notice_path="NOTICE",
        model_filename="model.onnx",
        model_size=1,
        model_sha256="sha256:" + "1" * 64,
        input_name="input",
        input_size=size,
        output_names=("mask_logits",),
        redistributable=False,
    )


def _logits(mask):
    foreground = np.where(mask == 1, 8.0, -8.0).astype(np.float32)
    return {"mask_logits": np.stack((-foreground, foreground), axis=0)[None]}


class Backend:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    def run(self, tensor):
        self.calls.append(tensor.copy())
        return self.outputs


def test_lazy_onnx_backend_does_not_open_session_until_first_inference():
    tensor = np.zeros((1, 3, 128, 128), np.float32)
    outputs = {"mask_logits": np.zeros((1, 2, 128, 128), np.float32)}

    with patch("photocut.algorithms.v8.provider.OnnxRuntimeBackend") as backend_type:
        backend_type.return_value.run.return_value = outputs
        backend = LazyOnnxRuntimeBackend(Path("model.onnx"), _manifest())

        backend_type.assert_not_called()
        assert backend.run(tensor) is outputs
        assert backend.run(tensor) is outputs

    backend_type.assert_called_once()
    assert backend_type.return_value.run.call_count == 2


def test_provider_uses_shared_preprocess_emits_stable_refined_candidate():
    image = np.full((80, 100, 3), 235, np.uint8)
    corners = ((10, 8), (91, 14), (84, 72), (15, 68))
    model_size = (128, 128)
    _, transform = letterbox_rgb_nchw(image, model_size)
    mask = rasterize_photo_mask(model_size, transform.to_model(corners))
    backend = Backend(_logits(mask))
    provider = PhotoMaskProvider(backend, _manifest(model_size))
    context = ImageFeatureContext(image)
    params = V7Parameters(scene_profile="scanner_white")

    first = provider.provide(context, params)
    second = provider.provide(context, params)

    assert first.status is ProviderStatus.SUCCESS
    assert len(first.candidates) == 1
    expected = preprocess_mask_input(image, model_size).tensor
    assert backend.calls[0].tobytes() == expected.tobytes()
    payload = first.to_dict()["candidates"][0]
    assert payload["candidate_id"] == second.to_dict()["candidates"][0]["candidate_id"]
    assert payload["source"] == "v8-test-mask:mask"
    assert payload["evidence"]["model_sha256"] == _manifest().model_sha256
    assert "raw_mask_corners" in payload["evidence"]
    assert "refinement" in payload["evidence"]


def test_provider_can_return_original_space_probability_without_second_inference():
    image = np.full((80, 100, 3), 235, np.uint8)
    corners = ((10, 8), (91, 14), (84, 72), (15, 68))
    model_size = (128, 128)
    _, transform = letterbox_rgb_nchw(image, model_size)
    mask = rasterize_photo_mask(model_size, transform.to_model(corners))
    backend = Backend(_logits(mask))
    provider = PhotoMaskProvider(backend, _manifest(model_size))

    result, probability = provider.provide_with_probability(
        ImageFeatureContext(image),
        V7Parameters(scene_profile="scanner_white"),
    )

    assert result.status is ProviderStatus.SUCCESS
    assert len(backend.calls) == 1
    assert probability.shape == image.shape[:2]
    assert probability.dtype == np.float32
    assert np.isfinite(probability).all()
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0
    assert float(probability[40, 50]) > 0.99
    assert float(probability[0, 0]) < 0.01


def test_probability_survives_when_full_frame_mask_is_too_large_to_refine():
    model_size = (768, 768)
    image = np.full((1536, 1536, 3), 235, np.uint8)
    backend = Backend(_logits(np.ones(model_size, np.uint8)))
    provider = PhotoMaskProvider(backend, _manifest(model_size))

    result, probability = provider.provide_with_probability(
        ImageFeatureContext(image),
        V7Parameters(scene_profile="scanner_white"),
    )

    assert result.status is ProviderStatus.NO_CANDIDATE
    assert not result.candidates
    assert result.to_dict()["diagnostics"]["reason"] == (
        "mask_candidate_refinement_invalid"
    )
    assert len(backend.calls) == 1
    assert probability.shape == image.shape[:2]
    assert float(probability.min()) > 0.99


def test_provider_never_runs_model_for_generic_single():
    backend = Backend({})
    provider = PhotoMaskProvider(backend, _manifest())
    result = provider.provide(
        ImageFeatureContext(np.zeros((20, 30, 3), np.uint8)),
        V7Parameters(scene_profile="generic_single"),
    )

    assert result.status is ProviderStatus.NO_CANDIDATE
    assert not backend.calls
    assert result.to_dict()["diagnostics"]["reason"] == "scene_profile_disabled"


def test_provider_reports_malformed_output_as_error_without_candidate():
    backend = Backend({"mask_logits": np.zeros((1, 1, 128, 128), np.float32)})
    provider = PhotoMaskProvider(backend, _manifest())

    result = provider.provide(
        ImageFeatureContext(np.zeros((80, 100, 3), np.uint8)),
        V7Parameters(scene_profile="scanner_white"),
    )

    assert result.status is ProviderStatus.ERROR
    assert not result.candidates
    assert result.error_code
