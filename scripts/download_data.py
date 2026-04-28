"""CLI: download Sentinel-2 + Dynamic World labels for a named AOI.

Prerequisites for the Dynamic World step (GOOGLE/DYNAMICWORLD/V1 via Earth Engine):
  1. One-time:  `earthengine authenticate`
  2. Every run: pass `--ee-project <your-gcp-project>` or set env var EE_PROJECT
     (or GOOGLE_CLOUD_PROJECT) to a Google Cloud project that has the Earth
     Engine API enabled.
"""
# CLI: command-line interface
# AOI: Area of interest

from __future__ import annotations
    # This makes type hints behave a bit more cleanly and consistently, especially for modern Python typing.

import argparse

from src.data.download import download_s2_and_dw


AOIS: dict[str, tuple[float, float, float, float]] = {
    # (west, south, east, north) in EPSG:4326
    "bay_area": (-122.55, 37.70, -122.35, 37.85),
    "napa":     (-122.45, 38.30, -122.20, 38.50),
    "sonoma":   (-122.85, 38.25, -122.60, 38.50),
    "lake_county_south": (-122.78, 38.70, -122.55, 38.90),
    "sac_urban": (-121.58, 38.51, -121.42, 38.63),
    "sac_wui": (-121.22, 38.60, -120.98, 38.76)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, choices=list(AOIS))
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-cloud", type=float, default=10.0)
    ap.add_argument(
        "--ee-project",
        default=None,
        help="Google Cloud project for Earth Engine (falls back to EE_PROJECT env var).",
    )
    args = ap.parse_args()

    bbox = AOIS[args.aoi]
    out_dir = args.out or f"data/raw/{args.aoi}"
    paths = download_s2_and_dw(
        bbox=bbox, start=args.start, end=args.end,
        out_dir=out_dir, max_cloud=args.max_cloud,
        ee_project=args.ee_project,
    )
    print("done:", paths)


if __name__ == "__main__":
    main()
