"""PyTorch Dataset for paired (image, mask) chips stored as .npy files."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


# Dynamic World / IO LULC mapping.
# DW classes: 0 water, 1 trees, 2 grass, 3 flooded_veg, 4 crops,
#             5 shrub_scrub, 6 built, 7 bare, 8 snow_ice
#
# Two label schemes are supported:
#   VEG_CLASSES  : any live vegetation → 1 (legacy / broad recall task)
#   WOODY_CLASSES: only woody fuels (trees + shrub/scrub) → 1
#                  This matches the wildfire-mitigation use case — grass and
#                  crops are cut / seasonal and are handled by a different
#                  program, but trees and shrubs drive the outage and ignition
#                  risk addressed by GO 95 / Rule 35 clearance standards.
#   WOODY_4CLASS : {other, trees, shrub_scrub, grass}
VEG_CLASSES = {1, 2, 3, 4, 5}
WOODY_CLASSES = {1, 5}
GRASS_CLASSES = {2}


def dw_to_woody_4class(msk: np.ndarray) -> np.ndarray:
    out = np.zeros_like(msk, dtype=np.int64)  # 0 = other
    out[msk == 1] = 1  # trees
    out[msk == 5] = 2  # shrub_scrub
    out[msk == 2] = 3  # grass
    return out


class ChipDataset(Dataset):
    """
    Reads paired (image, mask) chips written by `src.data.tiling.make_chips`.

    Parameters
    ----------
    chip_dir : directory containing `images/` and `masks/` subfolders
    band_indices : which band indices to keep from the image (e.g. (0,1,2) for RGB)
    task : 'binary'       -> collapse land-cover to any-vegetation vs rest
           'woody'        -> binarize to woody (trees+shrub) vs rest
           'woody_4class' -> 4-class map: other / trees / shrub / grass
           'multiclass'   -> keep original label integer
    transform : albumentations.Compose applied to (image, mask)
    normalize : whether to rescale bands to [0,1] using reflectance-ish constants
    prescaled : if True, chips are assumed to already be in per-band physical
                scales (e.g. output of build_feature_stack). Skips reflectance
                normalization.
    """

    def __init__(
        self,
        chip_dir: str | Path,
        band_indices: tuple[int, ...] = (2, 1, 0),  # RGB from S2 B04/B03/B02
        task: str = "binary",
        transform: Callable | None = None,
        normalize: bool = True,
        max_reflectance: float = 10000.0,
        prescaled: bool = False,
    ) -> None:
        self.chip_dir = Path(chip_dir)
        self.image_dir = self.chip_dir / "images"
        self.mask_dir = self.chip_dir / "masks"
        self.ids = sorted(p.stem for p in self.image_dir.glob("*.npy"))
        if not self.ids:
            raise RuntimeError(f"No chips found under {self.chip_dir}")

        self.band_indices = band_indices
        self.task = task
        self.transform = transform
        self.normalize = normalize
        self.max_reflectance = max_reflectance
        self.prescaled = prescaled

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        chip_id = self.ids[idx]
        img = np.load(self.image_dir / f"{chip_id}.npy")  # (C, H, W)
        msk = np.load(self.mask_dir / f"{chip_id}.npy")   # (H, W)

        img = img[list(self.band_indices), :, :]
        if self.prescaled:
            img = img.astype(np.float32)
        elif self.normalize:
            img = np.clip(img / self.max_reflectance, 0.0, 1.0).astype(np.float32)

        if self.task == "binary":
            msk = np.isin(msk, list(VEG_CLASSES)).astype(np.int64)
        elif self.task == "woody":
            # 1 = woody vegetation (trees + shrub/scrub), 0 = everything else.
            # This is the label that `scripts/build_risk_layer.py` consumes as
            # the "fuel" term in the downstream wildfire-risk calculation.
            msk = np.isin(msk, list(WOODY_CLASSES)).astype(np.int64)
        elif self.task == "woody_4class":
            # 0 = other, 1 = trees, 2 = shrub_scrub, 3 = grass.
            # This lets the model distinguish woody from low vegetation while
            # still supporting a merged woody probability at inference.
            msk = dw_to_woody_4class(msk)
        else:
            msk = msk.astype(np.int64)

        if self.transform is not None:
            # albumentations expects HWC
            out = self.transform(image=img.transpose(1, 2, 0), mask=msk)
            img = out["image"].transpose(2, 0, 1)
            msk = out["mask"]

        return torch.from_numpy(img).float(), torch.from_numpy(msk).long()
