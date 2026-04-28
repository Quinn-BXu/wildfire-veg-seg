"""
CLI: build the geographic feature layers for one AOI.

This script produces TWO kinds of artifacts:

  (A) Model input stack  → {raw_dir}/features.tif
      7 channels by default: [B02, B03, B04, B08, NDVI, sin_doy, cos_doy].
      Scaled, stackable, ready for `make_chips.py` and U-Net training.

  (B) Standalone GIS layers (NOT fed into the model):
        {raw_dir}/dem.tif          raw Copernicus DEM (meters)
        {raw_dir}/terrain.tif      elev_z, slope_deg/45, aspect_sin, aspect_cos
        {raw_dir}/distance.tif     Euclidean distance to OSM power=line, METERS

      These are consumed later by `scripts/build_risk_layer.py`, which combines
      them with the model's `woody_prob.tif` to produce the final wildfire-veg
      risk raster. Keeping them outside the model input avoids leakage between
      the "where's woody veg?" segmentation task and the downstream encroachment
      rule ("within X meters of a line").

Ablation extensions (opt-in):
    --include-terrain-in-stack      prepends the 4 terrain channels to features.tif
    --include-distance-in-stack     prepends the (scaled) distance channel
    --with-latlon-fourier           adds 4 * num_frequencies sinusoidal PE channels
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.geo_features import (
    build_feature_stack,
    build_ndvi_and_doy,
    compute_terrain,
    fetch_dem,
    fetch_powerlines_distance,
    latlon_fourier_channels,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/bay_area")
    ap.add_argument("--date", required=True, help="Representative date YYYY-MM-DD for DOY")
    ap.add_argument("--with-latlon-fourier", action="store_true")
    ap.add_argument("--fourier-freqs", type=int, default=4)
    ap.add_argument(
        "--include-terrain-in-stack",
        action="store_true",
        help="Append terrain (4ch) into features.tif (ablation only).",
    )
    ap.add_argument(
        "--include-distance-in-stack",
        action="store_true",
        help="Append scaled distance-to-powerline (1ch) into features.tif "
             "(ablation only; creates leakage with the downstream risk layer).",
    )
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    s2 = raw / "s2.tif"
    if not s2.exists():
        raise SystemExit(f"missing {s2}; run download_data.py first")

    print(">> 1/4 DEM + terrain (GIS layer)")
    dem = fetch_dem(s2, raw / "dem.tif")
    terrain = compute_terrain(dem, raw / "terrain.tif")

    print(">> 2/4 OSM distance to power=line, in meters (GIS layer)")
    distance = fetch_powerlines_distance(s2, raw / "distance.tif")

    print(">> 3/4 NDVI + DOY (goes into model stack)")
    ndvi_doy = build_ndvi_and_doy(s2, args.date, raw / "ndvi_doy.tif")

    fourier = None
    if args.with_latlon_fourier:
        print(">> 3b/4 lat/lon Fourier (optional model stack extension)")
        fourier = latlon_fourier_channels(
            s2, raw / "latlon_fourier.tif", num_frequencies=args.fourier_freqs
        )

    print(">> 4/4 stacking features.tif (model input)")
    build_feature_stack(
        s2_path=s2,
        ndvi_doy_path=ndvi_doy,
        out_path=raw / "features.tif",
        terrain_path=terrain if args.include_terrain_in_stack else None,
        distance_path=distance if args.include_distance_in_stack else None,
        latlon_fourier_path=fourier,
    )
    print("done.")
    print(f"  model input : {raw/'features.tif'}")
    print(f"  GIS layers  : {raw/'terrain.tif'}, {raw/'distance.tif'}, {raw/'dem.tif'}")


if __name__ == "__main__":
    main()
