"""Train/val augmentation pipelines — albumentations-based."""
from __future__ import annotations

import albumentations as A


def build_train_transform(chip_size: int = 256) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(size = (chip_size, chip_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(0.0, 0.002), p=0.2),
        ]
    )


def build_val_transform(chip_size: int = 256) -> A.Compose:
    return A.Compose(
        [A.CenterCrop(height=chip_size, width=chip_size, p=1.0)]
    )
