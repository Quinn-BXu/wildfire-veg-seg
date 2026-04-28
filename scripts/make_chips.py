"""CLI: tile a raw S2 + label pair into training chips."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.tiling import make_chips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/bay_area")
    ap.add_argument("--out-dir", default="data/processed/bay_area")
    ap.add_argument("--image-name", default="s2.tif",
                    help="use 'features.tif' for the multimodal stack")
    ap.add_argument("--chip-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--min-valid-frac", type=float, default=0.5)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    n = make_chips(
        image_path=raw / args.image_name,
        label_path=raw / "labels.tif",
        out_dir=args.out_dir,
        chip_size=args.chip_size,
        stride=args.stride,
        min_valid_frac=args.min_valid_frac,
    )
    print(f"wrote {n} chips")


if __name__ == "__main__":
    main()
