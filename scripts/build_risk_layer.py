"""
CLI: combine the model's woody-vegetation probability with GIS layers
(distance to power line + slope) into a final wildfire-vegetation risk raster.

This is the GIS half of the hybrid ML + GIS pipeline. The model is responsible
for *pattern recognition* ("is this pixel woody vegetation?"). This script is
responsible for *deterministic spatial rules* ("is that woody pixel close
enough to infrastructure and on steep enough terrain to matter for wildfire
mitigation?").

Formula
-------
        risk = woody_prob
             * exp(-distance_m / lambda_m)                 # proximity decay
             * clip(slope_deg / slope_scale_deg, 0, 1)     # terrain term

    All three factors are in [0, 1] so `risk` is also in [0, 1].

Defaults are grounded in the PG&E wildfire-mitigation domain:
    --lambda-m 30     : GO 95 / Rule 35 radial clearance for distribution
                        lines is ~4 ft trimmed to 12 ft during fire season.
                        30 m sets a soft boundary: at 30 m the proximity
                        factor = 1/e ≈ 0.37; by 90 m it is 0.05.
    --slope-scale-deg 30 : terrain steeper than ~30° is the usual threshold
                        above which CAL FIRE treats slope as a meaningful
                        accelerant for surface-to-crown fire spread.

Neither default is magic — they are commented so reviewers know they are
tunable knobs, not ML hyperparameters baked into the model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


def _read_single_band(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as ds:
        arr = ds.read(band).astype(np.float32)
        profile = ds.profile.copy()
    return arr, profile


def build_risk_layer(
    woody_prob_path: str | Path,
    distance_path: str | Path,
    terrain_path: str | Path,
    out_path: str | Path,
    lambda_m: float = 30.0,
    slope_scale_deg: float = 30.0,
    slope_terrain_band: int = 2,
    slope_terrain_scale: float = 45.0,
) -> Path:
    """
    Parameters
    ----------
    woody_prob_path     : single-band float32 raster in [0,1] (from predict.py)
    distance_path       : single-band float32 raster, meters (from geo_features)
    terrain_path        : 4-band float32 raster (elev_z, slope_n, asp_sin, asp_cos)
    out_path            : output .tif path (single-band float32 in [0,1])
    lambda_m            : decay length in meters for the proximity factor
    slope_scale_deg     : slope value (deg) at which the terrain factor saturates to 1
    slope_terrain_band  : which band of terrain.tif is the slope channel (1-indexed)
    slope_terrain_scale : divisor applied in compute_terrain() — used to recover
                          slope_deg from the normalized slope channel (defaults
                          match `slope_n = slope_deg / 45`).
    """
    woody, profile = _read_single_band(woody_prob_path, 1)
    distance_m, _ = _read_single_band(distance_path, 1)

    # recover slope in degrees from the normalized terrain channel
    slope_norm, _ = _read_single_band(terrain_path, slope_terrain_band)
    slope_deg = slope_norm * slope_terrain_scale

    if woody.shape != distance_m.shape or woody.shape != slope_deg.shape:
        raise ValueError(
            f"shape mismatch: woody={woody.shape}, dist={distance_m.shape}, "
            f"slope={slope_deg.shape}"
        )

    # --- components --- #
    prox = np.exp(-np.maximum(distance_m, 0.0) / lambda_m).astype(np.float32)
    slope_term = np.clip(slope_deg / slope_scale_deg, 0.0, 1.0).astype(np.float32)
    fuel = np.clip(woody, 0.0, 1.0).astype(np.float32)

    risk = (fuel * prox * slope_term).astype(np.float32)
    risk = np.clip(risk, 0.0, 1.0)

    # quick stats — useful when debugging tuning
    def _q(a, qs=(0.5, 0.9, 0.99)):
        return {f"p{int(q*100)}": float(np.quantile(a, q)) for q in qs}
    print(f"[risk] woody_prob mean={fuel.mean():.3f}, "
          f"proximity mean={prox.mean():.3f} "
          f"(lambda_m={lambda_m}), "
          f"slope mean={slope_term.mean():.3f} "
          f"(slope_scale_deg={slope_scale_deg})")
    print(f"[risk] output quantiles: {_q(risk)}  max={float(risk.max()):.3f}")

    profile.update(count=1, dtype="float32", compress="deflate", tiled=True, nodata=None)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(risk, 1)
    print(f"[risk] wrote {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--woody-prob", required=True, help="woody_prob.tif from predict.py")
    ap.add_argument("--distance", required=True, help="distance.tif in meters")
    ap.add_argument("--terrain", required=True, help="terrain.tif (4 bands)")
    ap.add_argument("--out", required=True, help="output risk.tif")
    ap.add_argument("--lambda-m", type=float, default=30.0,
                    help="decay length for proximity term (meters)")
    ap.add_argument("--slope-scale-deg", type=float, default=30.0,
                    help="slope (deg) at which terrain factor saturates to 1")
    ap.add_argument("--slope-terrain-band", type=int, default=2,
                    help="1-indexed band of terrain.tif containing slope")
    ap.add_argument("--slope-terrain-scale", type=float, default=45.0,
                    help="divisor used in compute_terrain to normalize slope")
    args = ap.parse_args()

    build_risk_layer(
        woody_prob_path=args.woody_prob,
        distance_path=args.distance,
        terrain_path=args.terrain,
        out_path=args.out,
        lambda_m=args.lambda_m,
        slope_scale_deg=args.slope_scale_deg,
        slope_terrain_band=args.slope_terrain_band,
        slope_terrain_scale=args.slope_terrain_scale,
    )


if __name__ == "__main__":
    main()
