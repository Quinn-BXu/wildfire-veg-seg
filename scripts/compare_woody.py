"""
CLI: compare two woody-vegetation rasters — typically the trained U-Net output
(woody_prob.tif) against the Dynamic World direct baseline (woody_prob_dw.tif).

Why this exists
---------------
The portfolio's core claim is "training a model buys something over using DW
labels directly". This script turns that claim into numbers. It reports:

  * Coverage            : fraction of valid pixels in each raster
  * Mean predicted prob : sanity check that neither side is degenerate
  * Agreement @ τ=0.5   : fraction of pixels where both rasters agree after
                          thresholding the continuous raster at τ
  * IoU (a vs b)        : Jaccard of the two binary masks — NOT ground truth,
                          just a measure of how much the model diverges from DW
  * Edge disagreement   : fraction of disagreement pixels that lie within N
                          pixels of a DW class boundary. High values here are
                          the good kind of disagreement: the model is cleaning
                          up DW's noisy canopy edges.
  * Optional ground-truth IoU against a third raster (e.g. a hand-labeled AOI
                          or a higher-resolution fuel map) via --gt.

Usage
-----
    python scripts/compare_woody.py \\
        --a outputs/bay_area/woody_prob.tif \\
        --b outputs/bay_area/woody_prob_dw.tif \\
        --threshold 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_dilation


def _read(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1).astype(np.float32)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else float("nan")


def compare(
    a_path: str | Path,
    b_path: str | Path,
    threshold: float = 0.5,
    edge_dilation_px: int = 2,
    gt_path: str | Path | None = None,
) -> dict:
    a = _read(a_path)
    b = _read(b_path)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a={a.shape}, b={b.shape}")

    a_bin = a >= threshold
    b_bin = b >= threshold

    agreement = float((a_bin == b_bin).mean())
    iou_ab = _iou(a_bin, b_bin)
    a_coverage = float(a_bin.mean())
    b_coverage = float(b_bin.mean())

    # edge disagreement: how much disagreement lives near DW class boundaries
    disagree = a_bin != b_bin
    b_edge = np.logical_xor(b_bin, binary_dilation(b_bin, iterations=edge_dilation_px))
    edge_mix = float((disagree & b_edge).sum() / max(disagree.sum(), 1))

    out = {
        "a": str(a_path),
        "b": str(b_path),
        "threshold": threshold,
        "coverage_a": a_coverage,
        "coverage_b": b_coverage,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "agreement": agreement,
        "iou_a_vs_b": iou_ab,
        "edge_disagreement_frac": edge_mix,
    }

    if gt_path is not None:
        gt = _read(gt_path)
        if gt.shape != a.shape:
            raise ValueError(f"gt shape {gt.shape} != {a.shape}")
        gt_bin = gt >= threshold
        out["iou_a_vs_gt"] = _iou(a_bin, gt_bin)
        out["iou_b_vs_gt"] = _iou(b_bin, gt_bin)

    print("[compare] results:")
    for k, v in out.items():
        if isinstance(v, float):
            print(f"  {k:>24s}: {v:.4f}")
        else:
            print(f"  {k:>24s}: {v}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first woody raster (usually model output)")
    ap.add_argument("--b", required=True, help="second woody raster (usually DW-direct)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability threshold for binarization")
    ap.add_argument("--edge-dilation-px", type=int, default=2,
                    help="dilation radius for the edge-disagreement mask")
    ap.add_argument("--gt", default=None,
                    help="optional ground-truth raster for IoU against both")
    args = ap.parse_args()
    compare(
        a_path=args.a,
        b_path=args.b,
        threshold=args.threshold,
        edge_dilation_px=args.edge_dilation_px,
        gt_path=args.gt,
    )


if __name__ == "__main__":
    main()
