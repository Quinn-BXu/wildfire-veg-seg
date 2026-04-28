"""
CLI: run a trained U-Net over an AOI's features.tif and write a probability
raster of woody vegetation (class 1).

Output:
    {out_path}   single-band float32 GeoTIFF in [0, 1],
                 same CRS + transform + shape as the input features.tif.

This raster is the first input to `scripts/build_risk_layer.py`, which
combines it with distance.tif and terrain.tif to form the final wildfire-veg
risk raster. For multiclass runs, the output is the merged woody probability:
P(trees) + P(shrub_scrub).

Design notes
------------
* Sliding window inference with 50% overlap + Hann-window blending. This
  removes the visible grid seams you get from hard tile boundaries and is
  robust to any single-tile miscalibration.
* Mixed precision is used automatically on CUDA.
* Input bands are taken from `cfg.data.band_indices` so the config that
  trained the model dictates exactly which channels are fed in at inference.
* For `task: woody_4class`, class indices 1 and 2 are merged into a single
  woody probability raster.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from omegaconf import OmegaConf

from src.models.unet import build_unet
from src.training.lightning_module import SegModule


def _hann_window_2d(size: int) -> np.ndarray:
    """2D Hann window for smooth tile blending."""
    w = np.hanning(size).astype(np.float32)
    return np.outer(w, w)


def _build_model_from_cfg(cfg) -> torch.nn.Module:
    if cfg.model.name != "unet":
        raise ValueError(f"unsupported model: {cfg.model.name}")
    return build_unet(
        encoder_name=cfg.model.encoder_name,
        encoder_weights=None,  # weights come from the checkpoint
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
    )


def _load_module(cfg, ckpt_path: str | Path, device: torch.device) -> SegModule:
    model = _build_model_from_cfg(cfg)
    module = SegModule.load_from_checkpoint(
        str(ckpt_path),
        model=model,
        num_classes=cfg.model.num_classes,
        map_location=device,
    )
    module.eval()
    module.to(device)
    return module


def _woody_prob_from_logits(logits: torch.Tensor, task: str) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    if task == "woody_4class":
        return probs[:, 1, :, :] + probs[:, 2, :, :]
    return probs[:, 1, :, :]


@torch.no_grad()
def predict_raster(
    features_path: str | Path,
    ckpt_path: str | Path,
    config_path: str | Path,
    out_path: str | Path,
    chip_size: int = 256,
    overlap: float = 0.5,
    batch_size: int = 8,
) -> Path:
    cfg = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = _load_module(cfg, ckpt_path, device)
    task = str(cfg.data.get("task", "woody"))

    band_indices = list(cfg.data.band_indices)
    prescaled = bool(cfg.data.get("prescaled", False))
    max_refl = float(cfg.data.get("max_reflectance", 10000.0))

    stride = max(1, int(chip_size * (1.0 - overlap)))
    hann = _hann_window_2d(chip_size)

    with rasterio.open(features_path) as ds:
        H, W = ds.height, ds.width
        profile = ds.profile.copy()
        bands = ds.read()  # (C, H, W)

    # pick + scale bands exactly as training did
    img = bands[band_indices, :, :].astype(np.float32)
    if not prescaled:
        img = np.clip(img / max_refl, 0.0, 1.0)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # gather tile coords (including an edge-aligned final row/col)
    ys = list(range(0, max(1, H - chip_size + 1), stride))
    if ys[-1] != H - chip_size:
        ys.append(max(0, H - chip_size))
    xs = list(range(0, max(1, W - chip_size + 1), stride))
    if xs[-1] != W - chip_size:
        xs.append(max(0, W - chip_size))

    coords = [(y, x) for y in ys for x in xs]
    print(f"[predict] {len(coords)} tiles ({chip_size}px, stride {stride}) "
          f"on {H}x{W} raster")

    use_amp = device.type == "cuda"
    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i : i + batch_size]
        batch = np.stack(
            [img[:, y : y + chip_size, x : x + chip_size] for (y, x) in batch_coords]
        )
        xb = torch.from_numpy(batch).to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = module(xb)
        probs = _woody_prob_from_logits(logits, task).float().cpu().numpy()

        for (y, x), p in zip(batch_coords, probs):
            prob_sum[y : y + chip_size, x : x + chip_size] += p * hann
            weight_sum[y : y + chip_size, x : x + chip_size] += hann

    # avoid div-by-zero at any untouched pixels (shouldn't happen with edge-aligned tiles)
    weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    prob = (prob_sum / weight_sum).astype(np.float32)
    prob = np.clip(prob, 0.0, 1.0)

    profile.update(count=1, dtype="float32", compress="deflate", tiled=True, nodata=None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(prob, 1)
    print(f"[predict] wrote {out_path}  (mean prob={prob.mean():.3f})")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config used for training")
    ap.add_argument("--ckpt", required=True, help="Lightning checkpoint path")
    ap.add_argument("--features", required=True, help="features.tif for the AOI")
    ap.add_argument("--out", required=True, help="output woody_prob.tif path")
    ap.add_argument("--chip-size", type=int, default=256)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    predict_raster(
        features_path=args.features,
        ckpt_path=args.ckpt,
        config_path=args.config,
        out_path=args.out,
        chip_size=args.chip_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
