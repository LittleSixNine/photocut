"""Lazy training-only LRASPP construction and loss for the V8 mask model."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def build_model(
    *,
    backbone_manifest: str | Path | None = None,
    backbone_path: str | Path | None = None,
) -> Any:
    """Construct the only V8 baseline architecture without automatic downloads."""
    import torch
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    if (backbone_manifest is None) != (backbone_path is None):
        raise ValueError("backbone_manifest and backbone_path must be supplied together")
    model = lraspp_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=2,
    )
    identity: dict[str, Any] = {"kind": "random_init"}
    if backbone_manifest is not None and backbone_path is not None:
        from .backbone_asset import load_backbone_manifest, verify_backbone_asset

        manifest_path = Path(backbone_manifest)
        manifest = load_backbone_manifest(manifest_path)
        verified = verify_backbone_asset(manifest, manifest_path.parent)
        if Path(backbone_path).resolve(strict=True) != verified:
            raise ValueError("backbone_path does not match verified manifest asset")
        state = torch.load(str(verified), map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("backbone asset is not a state dictionary")
        features = {
            key.removeprefix("features."): value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("features.")
        }
        model.backbone.load_state_dict(features, strict=True)
        identity = {
            "kind": "verified_pretrained",
            "asset_id": manifest.asset_id,
            "weight_enum": manifest.weight_enum,
            "weight_sha256": manifest.weight_sha256,
        }
    setattr(model, "_v8_backbone_identity", identity)
    return model


def backbone_identity(model: Any) -> dict[str, Any]:
    value = getattr(model, "_v8_backbone_identity", None)
    if not isinstance(value, dict):
        raise ValueError("model has no V8 backbone identity")
    return dict(value)


def export_logits(model: Any) -> Any:
    """Wrap TorchVision's mapping output as one ONNX-friendly logits tensor."""
    import torch

    class LogitsOnly(torch.nn.Module):
        def __init__(self, wrapped: Any):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, inputs):
            return self.wrapped(inputs)["out"]

    return LogitsOnly(model)


def segmentation_loss(logits: Any, target: Any) -> Any:
    """Foreground-weighted CE plus soft Dice with a fixed narrow boundary band."""
    import torch
    import torch.nn.functional as functional

    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(target, torch.Tensor)
        or logits.ndim != 4
        or logits.shape[1] != 2
        or target.ndim != 3
        or tuple(logits.shape[0:1] + logits.shape[2:]) != tuple(target.shape)
    ):
        raise ValueError("logits/target shapes must be N,2,H,W and N,H,W")
    if target.dtype != torch.long or not bool(torch.all((target == 0) | (target == 1))):
        raise ValueError("target must be a binary torch.long tensor")
    foreground = target.float().unsqueeze(1)
    dilated = functional.max_pool2d(foreground, kernel_size=5, stride=1, padding=2)
    eroded = -functional.max_pool2d(-foreground, kernel_size=5, stride=1, padding=2)
    boundary = (dilated - eroded).clamp(0.0, 1.0).squeeze(1)
    pixel_weight = 1.0 + 2.0 * boundary
    class_weight = torch.tensor((1.0, 2.0), dtype=logits.dtype, device=logits.device)
    cross_entropy = functional.cross_entropy(logits, target, weight=class_weight, reduction="none")
    weighted_ce = (cross_entropy * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)
    probability = torch.softmax(logits, dim=1)[:, 1]
    target_float = target.float()
    intersection = (probability * target_float).sum(dim=(1, 2))
    denominator = probability.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return weighted_ce + 0.5 * dice_loss


__all__ = [
    "backbone_identity",
    "build_model",
    "export_logits",
    "segmentation_loss",
]
