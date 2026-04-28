"""
CLI: convert Dynamic World labels directly to a woody "probability" raster,
with NO model involved. This is the honest baseline for the portfolio —
it answers the question "how much does the trained U-Net actually buy us
over just piping the DW labels straight into build_risk_layer.py?".

Output:
    {out_path}  single-band float32 GeoTIFF, same CRS + transform as labels.tif
                values ∈ {0.0, 1.0}   (hard labels; DW is categorical)

You can feed this file into scripts/build_risk_layer.py exactly the same way
you feed the model output, which lets us A/B the two pipelines with all
downstream rules held constant:

    python scripts/build_risk_layer.py \\
        --woody-prob outputs/bay_area/woody_prob_dw.tif \\
        --distance   data/raw/bay_area/distance.tif \\
        --terrain    data/raw/bay_area/terrain.tif \\
        --out        outputs/bay_area/risk_dw.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from src.data.dataset import WOODY_CLASSES


def dw_labels_to_woody_prob(
    labels_path: str | Path,
    out_path: str | Path,
    woody_classes: set[int] | None = None,
) -> Path:
    classes = woody_classes if woody_classes is not None else WOODY_CLASSES
    with rasterio.open(labels_path) as ds:
        lbl = ds.read(1)
        profile = ds.profile.copy()

    woody = np.isin(lbl, list(classes)).astype(np.float32)
    print(f"[dw-direct] woody fraction = {woody.mean():.3f}  "
          f"(classes={sorted(classes)})")

    profile.update(count=1, dtype="float32", compress="deflate", tiled=True, nodata=None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(woody, 1)
    print(f"[dw-direct] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="labels.tif from download_data.py")
    ap.add_argument("--out", required=True, help="output woody_prob_dw.tif path")
    args = ap.parse_args()
    dw_labels_to_woody_prob(args.labels, args.out)


if __name__ == "__main__":
    main()
