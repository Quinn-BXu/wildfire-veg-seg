"""Combined Dice + CrossEntropy loss, standard in segmentation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, ce_weight: torch.Tensor | None = None, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        dice_loss = _soft_dice_loss(logits, target)
        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss


def _soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    # ignore background (class 0) for the dice component
    return 1.0 - dice[1:].mean()
