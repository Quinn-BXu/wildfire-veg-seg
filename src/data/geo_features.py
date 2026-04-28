"""
Build geographic feature channels for the hybrid ML + GIS wildfire pipeline.

Two distinct products come out of this module:

A) Model input stack (features.tif) — default 7 channels:
        0..3  : S2 reflectance bands (B02, B03, B04, B08) — scaled to [0, 1]
        4     : NDVI                                       — scaled to [0, 1]
        5     : sin(2π · DOY / 365)                        — [-1, 1]
        6     : cos(2π · DOY / 365)                        — [-1, 1]

    The model learns from imagery + seasonality only. Terrain and distance-to-
    infrastructure are deliberately withheld so the segmentation task is pure
    visual reasoning (is this pixel woody vegetation?).

B) GIS layers (separate rasters, NOT fed to the model):
        terrain.tif  : (elev_z, slope_deg, aspect_sin, aspect_cos)
        distance.tif : Euclidean distance to OSM power=line, in METERS
        dem.tif      : raw Copernicus DEM (meters)

    These are combined with the model's predicted woody_prob.tif by
    `scripts/build_risk_layer.py` to produce the final wildfire-veg risk raster.

Optional model-stack extension (disabled by default):
    lat/lon Fourier features for explicit geographic positional encoding.
    See `latlon_fourier_channels` below.

All feature rasters are reprojected / resampled to the Sentinel-2 raster grid
(same CRS, transform, size) so they can be stacked band-wise into one GeoTIFF.

# Encode latitude/longitude with sinusoidal Fourier features rather than raw
# coordinates so the model sees smoother multi-scale spatial variation.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import distance_transform_edt
from shapely.geometry import LineString
from shapely.ops import transform as shp_transform
from pyproj import Transformer

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "text/plain; charset=utf-8",
    "User-Agent": "wildfire-veg-seg/0.1",
}

# band indices in the raw s2.tif (rasterio 1-indexed) given DEFAULT_S2_BANDS order
S2_BLUE = 1  # B02
S2_GREEN = 2  # B03
S2_RED = 3  # B04
S2_NIR = 4  # B08
MAX_REFLECTANCE = 10000.0


# ----------------------------- small utilities ----------------------------- #

def _open_catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _reference(ref_path: str | Path):
    with rasterio.open(ref_path) as ds:
        return dict(
            profile=ds.profile,
            transform=ds.transform,
            crs=ds.crs,
            width=ds.width,
            height=ds.height,
            bounds=ds.bounds,
        )


def _write(path: str | Path, data: np.ndarray, ref_profile: dict) -> Path:
    """Write (C, H, W) or (H, W) float32 array aligned to ref_profile grid."""
    if data.ndim == 2:
        data = data[None]
    profile = ref_profile.copy()
    profile.update(count=data.shape[0], dtype="float32", compress="deflate", tiled=True)
    with rasterio.open(path, "w", **profile) as dst:
        for i in range(data.shape[0]):
            dst.write(data[i].astype(np.float32), i + 1)
    return Path(path)


def _bbox_from_ref(ref_path: str | Path) -> tuple[float, float, float, float]:
    """Return ref bounds as (west, south, east, north) in EPSG:4326."""
    with rasterio.open(ref_path) as ds:
        bounds = ds.bounds
        crs = ds.crs
    if crs.to_string() == "EPSG:4326":
        return (bounds.left, bounds.bottom, bounds.right, bounds.top)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bounds.left, bounds.bottom)
    east, north = transformer.transform(bounds.right, bounds.top)
    return (west, south, east, north)


# ------------------------------ DEM + terrain ------------------------------ #

def fetch_dem(reference_raster: str | Path, out_path: str | Path) -> Path:
    """Fetch Copernicus DEM GLO-30, reproject into the reference grid."""
    ref = _reference(reference_raster)
    bbox = _bbox_from_ref(reference_raster)

    catalog = _open_catalog()
    items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
    if not items:
        raise RuntimeError("No Copernicus DEM items found for bbox")
    print(f"[dem] found {len(items)} tiles")

    dem_out = np.full((ref["height"], ref["width"]), np.nan, dtype=np.float32)
    for item in items:
        href = item.assets["data"].href
        with rasterio.open(href) as src:
            tmp = np.full((ref["height"], ref["width"]), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref["transform"],
                dst_crs=ref["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
            mask = ~np.isnan(tmp)
            dem_out[mask] = tmp[mask]

    # fill any remaining NaNs with the scene mean
    mean_val = np.nanmean(dem_out)
    dem_out = np.where(np.isnan(dem_out), mean_val, dem_out)
    return _write(out_path, dem_out, ref["profile"])


def compute_terrain(dem_path: str | Path, out_path: str | Path) -> Path:
    """Derive (elev_z, slope, aspect_sin, aspect_cos) from a DEM raster."""
    with rasterio.open(dem_path) as ds:
        dem = ds.read(1).astype(np.float32)
        transform = ds.transform
        profile = ds.profile
        crs = ds.crs

    # approximate pixel size in meters (S2 L2A is UTM → meters)
    px = abs(transform.a)
    if crs and crs.to_string() == "EPSG:4326":
        # degrees — rough conversion at the scene's mid-latitude
        mid_lat = (ds.bounds.bottom + ds.bounds.top) / 2
        px = px * 111_000.0 * np.cos(np.radians(mid_lat))

    dy, dx = np.gradient(dem, px)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    aspect_rad = np.arctan2(-dx, dy)  # 0 = north, +π/2 = east

    elev_z = (dem - np.nanmean(dem)) / (np.nanstd(dem) + 1e-6)
    slope_n = np.clip(slope_deg / 45.0, 0.0, 2.0)
    aspect_sin = np.sin(aspect_rad).astype(np.float32)
    aspect_cos = np.cos(aspect_rad).astype(np.float32)

    stack = np.stack([elev_z.astype(np.float32), slope_n.astype(np.float32),
                      aspect_sin, aspect_cos])
    return _write(out_path, stack, profile)


# ------------------ OSM power lines → distance raster ---------------------- #

def fetch_powerlines_distance(
    reference_raster: str | Path,
    out_path: str | Path,
    max_distance_m: float = 5000.0,
) -> Path:
    """
    Query OSM for power=line within the reference bbox, rasterize, distance transform.

    Output: single-band float32 GeoTIFF of Euclidean distance in METERS (clipped
    at max_distance_m for numerical safety). Raw meters are kept so downstream
    GIS scripts — e.g. risk-layer computation — can apply their own decay with
    interpretable constants (e.g. exp(-d / λ), λ=30m for GO 95 clearance zones).
    """
    ref = _reference(reference_raster)
    west, south, east, north = _bbox_from_ref(reference_raster)

    query = f"""
    [out:json][timeout:120];
    (
      way["power"="line"]({south},{west},{north},{east});
      way["power"="minor_line"]({south},{west},{north},{east});
      way["power"="cable"]({south},{west},{north},{east});
    );
    out geom;
    """
    try:
        r = requests.post(
            OVERPASS_URL,
            data=query.strip().encode("utf-8"),
            headers=OVERPASS_HEADERS,
            timeout=180,
        )
        r.raise_for_status()
        elements = r.json().get("elements", [])
    except Exception as e:
        print(f"[osm] warning: Overpass query failed ({e}); writing uniform distance")
        elements = []

    lines = [
        LineString([(pt["lon"], pt["lat"]) for pt in el["geometry"]])
        for el in elements
        if el.get("type") == "way" and len(el.get("geometry", [])) >= 2
    ]
    print(f"[osm] got {len(lines)} power-line ways")

    h, w = ref["height"], ref["width"]
    if not lines:
        # No lines within bbox — saturate everywhere at max_distance_m
        dist_m = np.full((h, w), max_distance_m, dtype=np.float32)
    else:
        transformer = Transformer.from_crs("EPSG:4326", ref["crs"], always_xy=True)
        lines_proj = [shp_transform(transformer.transform, ln) for ln in lines]
        mask = rasterize(
            [(ln, 1) for ln in lines_proj],
            out_shape=(h, w),
            transform=ref["transform"],
            fill=0,
            dtype=np.uint8,
        )
        dist_px = distance_transform_edt(mask == 0)
        px_size = abs(ref["transform"].a)  # meters in UTM
        dist_m = (dist_px * px_size).astype(np.float32)
        dist_m = np.clip(dist_m, 0.0, max_distance_m)

    return _write(out_path, dist_m, ref["profile"])


# ---------------------- NDVI + day-of-year channels ------------------------ #

def build_ndvi_and_doy(
    s2_path: str | Path,
    date_iso: str,
    out_path: str | Path,
) -> Path:
    """Compute NDVI + 2 constant DOY channels aligned to the S2 raster."""
    with rasterio.open(s2_path) as ds:
        red = ds.read(S2_RED).astype(np.float32)
        nir = ds.read(S2_NIR).astype(np.float32)
        profile = ds.profile
        h, w = ds.height, ds.width

    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    ndvi_scaled = ((ndvi + 1.0) / 2.0).astype(np.float32)  # -> [0,1]

    doy = datetime.fromisoformat(date_iso).timetuple().tm_yday
    phase = 2.0 * np.pi * doy / 365.0
    sin_doy = np.full((h, w), np.sin(phase), dtype=np.float32)
    cos_doy = np.full((h, w), np.cos(phase), dtype=np.float32)

    stack = np.stack([ndvi_scaled, sin_doy, cos_doy])
    return _write(out_path, stack, profile)


# --------------- optional: lat/lon Fourier feature channels ---------------- #

def latlon_fourier_channels(
    reference_raster: str | Path,
    out_path: str | Path,
    num_frequencies: int = 4,
) -> Path:
    """
    Per-pixel sinusoidal encoding of (lat, lon) at multiple frequencies.

    Produces 4 * num_frequencies channels:
      for k in [0, num_frequencies-1]:
        sin(π · 2^k · lat_norm), cos(π · 2^k · lat_norm),
        sin(π · 2^k · lon_norm), cos(π · 2^k · lon_norm)

    With num_frequencies=4 → 16 channels. This is plenty for most segmentation
    tasks; use more (e.g. 8) if you need continental-scale generalization.
    """
    ref = _reference(reference_raster)
    # Build lat/lon grid
    rows, cols = np.indices((ref["height"], ref["width"]))
    xs, ys = rasterio.transform.xy(ref["transform"], rows, cols, offset="center")
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if ref["crs"].to_string() != "EPSG:4326":
        transformer = Transformer.from_crs(ref["crs"], "EPSG:4326", always_xy=True)
        lons, lats = transformer.transform(xs, ys)
    else:
        lons, lats = xs, ys

    lat_n = (np.asarray(lats) / 90.0).astype(np.float32)
    lon_n = (np.asarray(lons) / 180.0).astype(np.float32)

    channels = []
    for k in range(num_frequencies):
        freq = 2.0 ** k
        channels.append(np.sin(np.pi * freq * lat_n))
        channels.append(np.cos(np.pi * freq * lat_n))
        channels.append(np.sin(np.pi * freq * lon_n))
        channels.append(np.cos(np.pi * freq * lon_n))

    stack = np.stack(channels).astype(np.float32)
    return _write(out_path, stack, ref["profile"])


# --------------------------- stacking + scaling ---------------------------- #

def build_feature_stack(
    s2_path: str | Path,
    ndvi_doy_path: str | Path,
    out_path: str | Path,
    terrain_path: str | Path | None = None,
    distance_path: str | Path | None = None,
    latlon_fourier_path: str | Path | None = None,
) -> Path:
    """
    Concatenate S2 (scaled to [0,1]) + derived rasters into the model-input GeoTIFF.

    Default output (7 channels):
        [B02, B03, B04, B08, NDVI, sin_doy, cos_doy]

    Optional extensions (off by default — kept for ablation experiments):
        terrain_path         → +4 channels (elev_z, slope, aspect_sin, aspect_cos)
        distance_path        → +1 channel  (distance to power line; NOT recommended
                               for the production model — leak between input and
                               the downstream encroachment mask)
        latlon_fourier_path  → +4·N channels of sinusoidal lat/lon positional encoding
    """
    with rasterio.open(s2_path) as ds:
        s2 = ds.read().astype(np.float32)  # (4, H, W)
        ref_profile = ds.profile
    s2_scaled = np.clip(s2 / MAX_REFLECTANCE, 0.0, 1.0)

    def _read(p):
        with rasterio.open(p) as d:
            return d.read().astype(np.float32)

    ndvi_doy = _read(ndvi_doy_path)    # (3, H, W): ndvi, sin_doy, cos_doy

    ordered = [
        s2_scaled,          # 0..3
        ndvi_doy[:1],       # 4 (NDVI)
        ndvi_doy[1:],       # 5..6 (sin_doy, cos_doy)
    ]
    if terrain_path is not None:
        ordered.append(_read(terrain_path))          # +4
    if distance_path is not None:
        # If distance is raw meters we scale to [0,1] here to keep model inputs
        # on a common scale (does not change the raw distance.tif GIS layer).
        dist = _read(distance_path)
        dist = np.clip(dist / 5000.0, 0.0, 1.0)
        ordered.append(dist)                         # +1
    if latlon_fourier_path is not None:
        ordered.append(_read(latlon_fourier_path))   # +4*N

    stack = np.concatenate(ordered, axis=0)
    print(f"[stack] model feature tensor: {stack.shape} "
          f"(terrain={'yes' if terrain_path else 'no'}, "
          f"distance={'yes' if distance_path else 'no'}, "
          f"latlon_fourier={'yes' if latlon_fourier_path else 'no'})")
    return _write(out_path, stack, ref_profile)
