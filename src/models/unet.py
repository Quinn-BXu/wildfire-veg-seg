"""Thin wrapper around segmentation_models_pytorch U-Net."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    num_classes: int = 2,
) -> nn.Module:
    """
    Returns an SMP U-Net. For binary tasks use num_classes=2 (softmax) — it gives
    cleaner metric code than a single-channel sigmoid.
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
    )
