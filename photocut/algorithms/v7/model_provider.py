"""Explicit, additive V7 candidate provider backed by a local model."""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any

from .features import FeatureCancelled, ImageFeatureContext
from .model_inference import InferenceBackend, letterbox_rgb_nchw
from .model_manifest import ModelManifest
from .model_postprocess import decode_docaligner_outputs, decode_docquad_outputs
from .parameters import V7Parameters
from .types import ProviderResult, ProviderStatus


def _cancelled(token: Any) -> bool:
    if token is None:
        return False
    for name in ("is_cancelled", "cancelled", "is_set"):
        value = getattr(token, name, None)
        if value is None:
            continue
        try:
            return bool(value() if callable(value) else value)
        except TypeError:
            continue
    return False


def _expired(deadline: Any) -> bool:
    if deadline is None:
        return False
    if isinstance(deadline, bool):
        return deadline
    value = deadline() if callable(deadline) else deadline
    if isinstance(value, bool):
        return value
    try:
        return float(value) <= time.monotonic()
    except (TypeError, ValueError):
        return False


class ModelQuadProvider:
    """Construct directly only; detector defaults never instantiate this class."""

    def __init__(self, backend: InferenceBackend, manifest: ModelManifest):
        self.backend = backend
        self.manifest = manifest
        self.name = manifest.adapter

    def provide(self, context: ImageFeatureContext, params: V7Parameters,
                cancellation_token: Any = None, deadline: Any = None) -> ProviderResult:
        started = time.perf_counter()
        work_limit = int(params.provider_work_limits.get(self.name, 120_000))
        used = 0
        try:
            if _cancelled(cancellation_token):
                raise FeatureCancelled("model provider cancelled")
            if _expired(deadline):
                raise TimeoutError("model provider deadline exceeded")
            image = context.image
            height, width = image.shape[:2]
            tensor, transform = letterbox_rgb_nchw(image, self.manifest.input_size)
            used += max(1, int(math.ceil(tensor.size / 4096.0)))
            if used > work_limit:
                raise MemoryError("model provider work budget exhausted")
            outputs = self.backend.run(tensor)
            if _cancelled(cancellation_token):
                raise FeatureCancelled("model provider cancelled")
            if _expired(deadline):
                raise TimeoutError("model provider deadline exceeded")
            if self.manifest.adapter == "docquadnet":
                proposals = decode_docquad_outputs(outputs, transform, (width, height))
            elif self.manifest.adapter == "docaligner_heatmap":
                proposals = decode_docaligner_outputs(outputs, transform, (width, height))
            else:  # guarded by manifest, retained for defensive isolation
                raise ValueError(f"unsupported model adapter: {self.manifest.adapter}")
            candidates = []
            for proposal in proposals:
                payload = json.dumps([
                    self.manifest.model_id, self.manifest.model_sha256, proposal.head,
                    [[round(float(x), 6), round(float(y), 6)] for x, y in proposal.corners],
                ], sort_keys=True, separators=(",", ":"))
                candidate_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
                evidence = dict(proposal.evidence)
                evidence.update({"model_id": self.manifest.model_id,
                                 "model_adapter": self.manifest.adapter,
                                 "model_manifest_sha256": self.manifest.model_sha256,
                                 "model_head": proposal.head})
                candidates.append({
                    "id": candidate_id, "candidate_id": candidate_id,
                    "corners": proposal.corners, "sources": (self.name, proposal.head),
                    "source": f"{self.manifest.model_id}:{proposal.head}",
                    "evidence": evidence, "score": float(proposal.confidence),
                })
            status = ProviderStatus.SUCCESS if candidates else ProviderStatus.NO_CANDIDATE
            return ProviderResult(self.name, tuple(candidates), status=status,
                                  elapsed_ms=(time.perf_counter() - started) * 1000,
                                  work_consumed=min(used, work_limit), work_limit=work_limit,
                                  diagnostics={"candidate_count": len(candidates),
                                               "model_id": self.manifest.model_id})
        except FeatureCancelled as exc:
            return ProviderResult(self.name, (), ProviderStatus.CANCELLED,
                                  elapsed_ms=(time.perf_counter() - started) * 1000,
                                  work_consumed=min(used, work_limit), work_limit=work_limit,
                                  error_code="cancelled", diagnostics={"message": str(exc)})
        except TimeoutError as exc:
            return ProviderResult(self.name, (), ProviderStatus.TIMEOUT,
                                  elapsed_ms=(time.perf_counter() - started) * 1000,
                                  work_consumed=min(used, work_limit), work_limit=work_limit,
                                  timeout_code="provider_deadline", diagnostics={"message": str(exc)})
        except MemoryError as exc:
            return ProviderResult(self.name, (), ProviderStatus.BUDGET_EXHAUSTED,
                                  elapsed_ms=(time.perf_counter() - started) * 1000,
                                  work_consumed=work_limit, work_limit=work_limit,
                                  error_code="work_limit", diagnostics={"message": str(exc)})
        except Exception as exc:
            return ProviderResult(self.name, (), ProviderStatus.ERROR,
                                  elapsed_ms=(time.perf_counter() - started) * 1000,
                                  work_consumed=min(used, work_limit), work_limit=work_limit,
                                  error_code=type(exc).__name__, diagnostics={"message": str(exc)})

    run = provide
    generate = provide
