"""Tile a large (image, label) raster pair into training chips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def make_chips(
    image_path: str | Path,
    label_path: str | Path,
    out_dir: str | Path,
    chip_size: int = 256,
    stride: int = 128,
    min_valid_frac: float = 0.5,
) -> int:
    """Slide a window over (image, label), write paired .npy chips.

    Returns the number of chips written.
    """
    out_dir = Path(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    n_written = 0
    with rasterio.open(image_path) as img_ds, rasterio.open(label_path) as lbl_ds:
        if (img_ds.height, img_ds.width) != (lbl_ds.height, lbl_ds.width):
            raise ValueError("image and label rasters must share shape (reproject first)")

        H, W = img_ds.height, img_ds.width
        for y in range(0, H - chip_size + 1, stride):
            for x in range(0, W - chip_size + 1, stride):
                window = Window(x, y, chip_size, chip_size)
                img = img_ds.read(window=window)      # (C, H, W)
                lbl = lbl_ds.read(1, window=window)   # (H, W)

                # skip chips with too much nodata
                valid = np.isfinite(img) & (img > 0)
                if valid.mean() < min_valid_frac:
                    continue

                chip_id = f"chip_{y:06d}_{x:06d}"
                np.save(out_dir / "images" / f"{chip_id}.npy", img.astype(np.float32))
                np.save(out_dir / "masks" / f"{chip_id}.npy", lbl.astype(np.int64))
                n_written += 1

    print(f"[tiling] wrote {n_written} chips to {out_dir}")
    return n_written
