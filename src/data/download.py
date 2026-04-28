"""
Download Sentinel-2 L2A imagery (Microsoft Planetary Computer) + Dynamic World
land-cover labels (Google Earth Engine) for a given AOI.

Two data sources are used on purpose:

    * Sentinel-2 L2A   ← Microsoft Planetary Computer STAC API (no auth)
    * Dynamic World v1 ← Google Earth Engine (requires `earthengine authenticate`
                         and a Google Cloud project with the Earth Engine API
                         enabled). Dynamic World is Google-published and is not
                         hosted on Planetary Computer under its real name — the
                         only honest way to say "Dynamic World labels" is to
                         pull them from Earth Engine.

Typical usage:

    from src.data.download import download_s2_and_dw
    download_s2_and_dw(
        bbox=(-122.55, 37.70, -122.35, 37.85),  # SF bay area
        start="2024-06-01",
        end="2024-09-30",
        out_dir="data/raw/bay_area",
        max_cloud=10,
        ee_project="my-gcp-project",            # or set EE_PROJECT env var
    )

Bands returned (Sentinel-2 L2A, 10m/20m):
    B02 (blue), B03 (green), B04 (red), B08 (nir) at 10m
    B05, B06, B07, B8A, B11, B12 at 20m (resampled)

Dynamic World v1 classes (0-8):
    0 water, 1 trees, 2 grass, 3 flooded_veg, 4 crops,
    5 shrub_scrub, 6 built, 7 bare, 8 snow_ice
"""
from __future__ import annotations

import json # used for metadata and file paths.
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
import stackstac
import xarray as xr
from rasterio.warp import Resampling, reproject


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

DEFAULT_S2_BANDS: tuple[str, ...] = ("B02", "B03", "B04", "B08")  # blue, green, red, nir


def _open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

def _search_items(
    catalog: pystac_client.Client,
    collection: str,
    bbox: Sequence[float],
    start: str,
    end: str,
    query: dict | None = None,
):
    return list(
        catalog.search(
            collections=[collection],
            bbox=bbox,
            datetime=f"{start}/{end}",
            query=query or {},
        ).items()
    )


def download_sentinel2(
    bbox: Sequence[float],
    start: str,
    end: str,
    out_path: str | Path,
    bands: Iterable[str] = DEFAULT_S2_BANDS,
    max_cloud: float = 10.0,
    resolution: int = 10,
    composite: str = "median",
) -> Path:
    """Download a Sentinel-2 L2A cloud-free median composite to a single GeoTIFF."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = _open_catalog()
    items = _search_items(
        catalog,
        "sentinel-2-l2a",
        bbox,
        start,
        end,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    if not items:
        raise RuntimeError(f"No S2 items for bbox={bbox} {start}-{end} cc<{max_cloud}")
    print(f"[s2] found {len(items)} items")

    stack = stackstac.stack(
        items,
        assets=list(bands),
        epsg=32610,
        resolution=resolution,
        bounds_latlon=tuple(bbox),
        chunksize=1024,
        rescale=False,
    )
    # Mask nodata, reduce temporal axis
    stack = stack.where(stack > 0)
    if composite == "median":
        composite_arr = stack.median(dim="time", skipna=True)
    elif composite == "mean":
        composite_arr = stack.mean(dim="time", skipna=True)
    else:
        raise ValueError(f"unknown composite {composite}")

    composite_arr = composite_arr.compute().astype("uint16")
    _write_raster(composite_arr, out_path, band_names=list(bands))
    print(f"[s2] wrote {out_path}")
    return out_path


# --------------------------- Dynamic World (GEE) --------------------------- #

def _init_earth_engine(project: str | None = None):
    """
    Lazy-import and initialize Earth Engine. Raises a clear error if the
    environment is not set up.

    Auth:
      1. One-time:  `earthengine authenticate`
      2. Every run: either pass `project=` or set env var EE_PROJECT (or
         GOOGLE_CLOUD_PROJECT) to a Google Cloud project with the Earth Engine
         API enabled.
    """
    try:
        import ee  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "earthengine-api is required for Dynamic World. "
            "`pip install earthengine-api` (already in requirements.txt)."
        ) from exc

    project = project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run:\n"
            "  earthengine authenticate\n"
            "then set EE_PROJECT to a GCP project with the Earth Engine API enabled,\n"
            "or pass --ee-project on the CLI."
        ) from exc
    return ee


def download_dynamic_world(
    bbox: Sequence[float],
    start: str,
    end: str,
    out_path: str | Path,
    reference_path: str | Path | None = None,
    scale: int = 10,
    ee_project: str | None = None,
) -> Path:
    """
    Download a Dynamic World v1 mode composite (per-pixel most-frequent class)
    via Google Earth Engine.

    If `reference_path` is given (e.g. the S2 raster for this AOI), the output
    is reprojected onto that exact grid so `make_chips.py` can pair them
    pixel-for-pixel without extra steps.
    """
    ee = _init_earth_engine(ee_project)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    west, south, east, north = bbox
    geom = ee.Geometry.BBox(west, south, east, north)

    col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(start, end)
        .filterBounds(geom)
    )
    n = col.size().getInfo()
    if n == 0:
        raise RuntimeError(f"No Dynamic World scenes for bbox={bbox} {start}-{end}")
    print(f"[dw] found {n} scenes")

    # Per-pixel mode of the categorical `label` band → single-band uint8 raster.
    mode_img = col.select("label").reduce(ee.Reducer.mode()).toUint8()

    # Match the reference S2 CRS if provided so the reprojection at the end is minimal.
    if reference_path is not None:
        with rasterio.open(reference_path) as ref:
            crs_str = ref.crs.to_string()
    else:
        crs_str = "EPSG:32610"

    url = mode_img.getDownloadURL(
        {
            "region": geom,
            "scale": scale,
            "crs": crs_str,
            "format": "GEO_TIFF",
        }
    )

    # Stream the GeoTIFF to a temp file, then align to the reference grid if given.
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        resp = requests.get(url, timeout=600)
        resp.raise_for_status()
        tmp.write(resp.content)
        tmp_path = Path(tmp.name)

    try:
        if reference_path is not None:
            _reproject_labels_to_ref(tmp_path, reference_path, out_path)
        else:
            tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    print(f"[dw] wrote {out_path}")
    return out_path


def _reproject_labels_to_ref(
    src_path: str | Path,
    ref_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Nearest-neighbor reproject a categorical raster onto the reference grid."""
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_h = ref.height
        ref_w = ref.width

    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_transform = src.transform
        src_crs = src.crs

    dst_arr = np.zeros((ref_h, ref_w), dtype=np.uint8)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest,  # categorical labels → nearest only
    )

    profile = {
        "driver": "GTiff",
        "height": ref_h,
        "width": ref_w,
        "count": 1,
        "dtype": "uint8",
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dst_arr, 1)
        dst.update_tags(**{"band_1": "dynamic_world_label"})
    return Path(out_path)


# ------------------------------ raster writer ------------------------------ #

def _write_raster(da: xr.DataArray, path: Path, band_names: list[str]) -> None:
    """Write a (band, y, x) or (y, x) xarray DataArray to a GeoTIFF."""
    if "band" not in da.dims:
        da = da.expand_dims("band")
    arr = da.values
    transform = da.rio.transform() if hasattr(da, "rio") else da.attrs.get("transform")
    crs = da.rio.crs if hasattr(da, "rio") else da.attrs.get("crs")
    if transform is None or crs is None:
        # stackstac output: pull from coords
        from rasterio.transform import from_bounds
        y = da["y"].values
        x = da["x"].values
        transform = from_bounds(x.min(), y.min(), x.max(), y.max(), len(x), len(y))
        crs = da.attrs.get("crs")
        if crs is None:
            raise ValueError("CRS missing; cannot safely write raster.")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[-2],
        width=arr.shape[-1],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
    ) as dst:
        for i in range(arr.shape[0]):
            dst.write(arr[i], i + 1)
        dst.update_tags(**{f"band_{i+1}": b for i, b in enumerate(band_names)})


# ------------------------------ convenience -------------------------------- #

def download_s2_and_dw(
    bbox: Sequence[float],
    start: str,
    end: str,
    out_dir: str | Path,
    max_cloud: float = 10.0,
    bands: Iterable[str] = DEFAULT_S2_BANDS,
    ee_project: str | None = None,
) -> dict[str, Path]:
    """Convenience wrapper — downloads both S2 (Planetary Computer) and
    Dynamic World labels (Earth Engine) for one AOI, aligned on the same grid."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s2_path = download_sentinel2(
        bbox, start, end, out_dir / "s2.tif", bands=bands, max_cloud=max_cloud
    )
    # Pass s2_path as the reference so labels.tif is reprojected to match.
    lc_path = download_dynamic_world(
        bbox, start, end, out_dir / "labels.tif",
        reference_path=s2_path, ee_project=ee_project,
    )

    meta = {
        "bbox": list(bbox),
        "start": start,
        "end": end,
        "bands": list(bands),
        "max_cloud": max_cloud,
        "label_source": "GOOGLE/DYNAMICWORLD/V1 (Earth Engine)",
        "label_classes": {
            "0": "water", "1": "trees", "2": "grass", "3": "flooded_veg",
            "4": "crops", "5": "shrub_scrub", "6": "built", "7": "bare",
            "8": "snow_ice",
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"s2": s2_path, "labels": lc_path}

'''
python -m scripts.download_data --aoi bay_area --start 2024-06-01 --end 2024-09-30
python -m scripts.build_geo_features --raw-dir data/raw/bay_area --date 2024-07-15
python -m scripts.make_chips --raw-dir data/raw/bay_area --out-dir data/processed/bay_area_woody --image-name features.tif
'''