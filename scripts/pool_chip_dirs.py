"""Pool multiple chip datasets into one directory with AOI-prefixed filenames."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_chip_set(src_dir: Path, out_dir: Path, prefix: str) -> tuple[int, int]:
    img_src = src_dir / "images"
    msk_src = src_dir / "masks"
    img_out = out_dir / "images"
    msk_out = out_dir / "masks"

    if not img_src.exists() or not msk_src.exists():
        raise FileNotFoundError(f"missing images/ or masks/ under {src_dir}")

    img_out.mkdir(parents=True, exist_ok=True)
    msk_out.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_msk = 0

    for img_path in sorted(img_src.glob("*.npy")):
        out_name = f"{prefix}_{img_path.name}"
        shutil.copy2(img_path, img_out / out_name)
        n_img += 1

    for msk_path in sorted(msk_src.glob("*.npy")):
        out_name = f"{prefix}_{msk_path.name}"
        shutil.copy2(msk_path, msk_out / out_name)
        n_msk += 1

    return n_img, n_msk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("PREFIX", "DIR"),
        required=True,
        help="source chip dataset as: --source bay_area data/processed/bay_area_woody",
    )
    ap.add_argument("--out-dir", required=True, help="pooled output directory")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    total_img = 0
    total_msk = 0

    for prefix, src in args.source:
        n_img, n_msk = copy_chip_set(Path(src), out_dir, prefix)
        total_img += n_img
        total_msk += n_msk
        print(f"[pool] {prefix}: copied {n_img} images, {n_msk} masks")

    print(f"[pool] total: {total_img} images, {total_msk} masks")
    print(f"[pool] wrote pooled dataset to {out_dir}")


if __name__ == "__main__":
    main()
