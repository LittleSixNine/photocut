"""Explicit scanner-white V8 mask candidate provider; never registered by default."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from threading import Lock
import time
from typing import Any

from photocut.algorithms.v7.features import FeatureCancelled, ImageFeatureContext
from photocut.algorithms.v7.model_inference import InferenceBackend, OnnxRuntimeBackend
from photocut.algorithms.v7.model_manifest import ModelManifest
from photocut.algorithms.v7.parameters import V7Parameters
from photocut.algorithms.v7.refinement import refine_quad
from photocut.algorithms.v7.types import ProviderResult, ProviderStatus

from .mask_postprocess import decode_photo_mask, restore_photo_mask_probability
from .preprocess import preprocess_mask_input


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "cancelled", "is_set"):
        value = getattr(token, name, None)
        if value is None:
            continue
        try:
            return bool(value() if callable(value) else value)
        except (TypeError, ValueError):
            continue
    return False


def _expired(deadline: Any) -> bool:
    if deadline is None:
        return False
    value = deadline() if callable(deadline) else deadline
    if isinstance(value, bool):
        return value
    try:
        return float(value) <= time.monotonic()
    except (TypeError, ValueError):
        return False


class LazyOnnxRuntimeBackend:
    """Open the optional V8 ONNX session only when mask inference starts."""

    def __init__(self, verified_model: Path, manifest: ModelManifest):
        self.verified_model = Path(verified_model)
        self.manifest = manifest
        self._backend: OnnxRuntimeBackend | None = None
        self._lock = Lock()

    def run(self, tensor: Any) -> Any:
        backend = self._backend
        if backend is None:
            with self._lock:
                backend = self._backend
                if backend is None:
                    backend = OnnxRuntimeBackend(
                        self.verified_model,
                        self.manifest,
                    )
                    self._backend = backend
        return backend.run(tensor)


class PhotoMaskProvider:
    """Construct directly for V8 shadow work; V7/auto never instantiate it."""

    name = "photo_mask_v8"

    def __init__(self, backend: InferenceBackend, manifest: ModelManifest, *, threshold: float = 0.5):
        if not isinstance(manifest, ModelManifest) or manifest.adapter != self.name:
            raise ValueError("PhotoMaskProvider requires a photo_mask_v8 manifest")
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0.05 <= float(threshold) <= 0.95:
            raise ValueError("mask threshold is out of bounds")
        self.backend = backend
        self.manifest = manifest
        self.threshold = float(threshold)

    def _provide_with_probability(
        self,
        context: ImageFeatureContext,
        params: V7Parameters,
        cancellation_token: Any = None,
        deadline: Any = None,
    ) -> tuple[ProviderResult, Any]:
        started = time.perf_counter()
        work_limit = int(params.provider_work_limits.get(self.name, 120_000))
        used = 0
        if params.scene_profile != "scanner_white":
            return ProviderResult(
                self.name,
                (),
                ProviderStatus.NO_CANDIDATE,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                work_limit=work_limit,
                diagnostics={"reason": "scene_profile_disabled"},
            ), None
        try:
            if _cancelled(cancellation_token):
                raise FeatureCancelled("V8 mask provider cancelled")
            if _expired(deadline):
                raise TimeoutError("V8 mask provider deadline exceeded")
            image = context.image
            height, width = image.shape[:2]
            prepared = preprocess_mask_input(image, self.manifest.input_size)
            used = max(1, int(math.ceil(prepared.tensor.size / 4096.0)))
            if used > work_limit:
                raise MemoryError("V8 mask provider work budget exhausted")
            outputs = self.backend.run(prepared.tensor)
            if _cancelled(cancellation_token):
                raise FeatureCancelled("V8 mask provider cancelled")
            if _expired(deadline):
                raise TimeoutError("V8 mask provider deadline exceeded")
            probability = restore_photo_mask_probability(
                outputs, prepared.transform
            )
            proposal = decode_photo_mask(
                outputs,
                prepared.transform,
                image_size=(width, height),
                threshold=self.threshold,
            )
            if proposal is None:
                return ProviderResult(
                    self.name,
                    (),
                    ProviderStatus.NO_CANDIDATE,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    work_consumed=used,
                    work_limit=work_limit,
                    diagnostics={"reason": "mask_has_no_legal_primary_component"},
                ), probability
            stable_payload = {
                "model_id": self.manifest.model_id,
                "model_sha256": self.manifest.model_sha256,
                "threshold": self.threshold,
                "corners": [[round(float(x), 6), round(float(y), 6)] for x, y in proposal.corners],
            }
            candidate_id = hashlib.sha256(
                json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:20]
            raw_candidate = {
                "id": candidate_id,
                "candidate_id": candidate_id,
                "corners": proposal.corners,
                "sources": (self.name,),
                "source": f"{self.manifest.model_id}:mask",
            }
            try:
                refinement = refine_quad(
                    context,
                    raw_candidate,
                    params,
                    cancellation_token=cancellation_token,
                    deadline=deadline,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                return ProviderResult(
                    self.name,
                    (),
                    ProviderStatus.NO_CANDIDATE,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    work_consumed=used,
                    work_limit=work_limit,
                    diagnostics={
                        "reason": "mask_candidate_refinement_invalid",
                        "message": str(exc),
                    },
                ), probability
            evidence = {
                **dict(proposal.evidence),
                "model_id": self.manifest.model_id,
                "model_sha256": self.manifest.model_sha256,
                "model_adapter": self.manifest.adapter,
                "raw_mask_corners": proposal.corners,
                "proposed_refined_corners": refinement.proposed_corners,
                "adopted_refined_corners": refinement.adopted_corners,
                "refinement": dict(refinement.evidence),
            }
            candidate = {
                **raw_candidate,
                "corners": refinement.adopted_corners,
                "raw_corners": proposal.corners,
                "score": proposal.confidence,
                "evidence": evidence,
            }
            return ProviderResult(
                self.name,
                (candidate,),
                ProviderStatus.SUCCESS,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                work_consumed=used,
                work_limit=work_limit,
                diagnostics={
                    "candidate_count": 1,
                    "model_id": self.manifest.model_id,
                    "refinement_adopted": refinement.adopted,
                },
            ), probability
        except FeatureCancelled as exc:
            return ProviderResult(
                self.name,
                (),
                ProviderStatus.CANCELLED,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                work_consumed=min(used, work_limit),
                work_limit=work_limit,
                error_code="cancelled",
                diagnostics={"message": str(exc)},
            ), None
        except TimeoutError as exc:
            return ProviderResult(
                self.name,
                (),
                ProviderStatus.TIMEOUT,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                work_consumed=min(used, work_limit),
                work_limit=work_limit,
                timeout_code="provider_deadline",
                diagnostics={"message": str(exc)},
            ), None
        except MemoryError as exc:
            return ProviderResult(
                self.name,
                (),
                ProviderStatus.BUDGET_EXHAUSTED,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                work_consumed=work_limit,
                work_limit=work_limit,
                error_code="work_limit",
                diagnostics={"message": str(exc)},
            ), None
        except Exception as exc:
            return ProviderResult(
                self.name,
                (),
                ProviderStatus.ERROR,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                work_consumed=min(used, work_limit),
                work_limit=work_limit,
                error_code=type(exc).__name__,
                diagnostics={"message": str(exc)},
            ), None

    def provide(
        self,
        context: ImageFeatureContext,
        params: V7Parameters,
        cancellation_token: Any = None,
        deadline: Any = None,
    ) -> ProviderResult:
        result, _probability = self._provide_with_probability(
            context,
            params,
            cancellation_token=cancellation_token,
            deadline=deadline,
        )
        return result

    def provide_with_probability(
        self,
        context: ImageFeatureContext,
        params: V7Parameters,
        cancellation_token: Any = None,
        deadline: Any = None,
    ) -> tuple[ProviderResult, Any]:
        """Return the candidate and its probability map from one model call."""
        return self._provide_with_probability(
            context,
            params,
            cancellation_token=cancellation_token,
            deadline=deadline,
        )

    run = provide
    generate = provide


__all__ = ["LazyOnnxRuntimeBackend", "PhotoMaskProvider"]
