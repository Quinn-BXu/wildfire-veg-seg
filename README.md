# Wildfire Vegetation Segmentation

> **Status:** Portfolio / research code. Single-author project intended to demonstrate end-to-end geospatial ML for wildfire vegetation risk; not production-hardened.

**Goal:** Identify woody vegetation that is both close to electric distribution infrastructure *and* on terrain that favors fire spread, using Sentinel-2 imagery plus open GIS layers.

The pipeline is deliberately a hybrid — a deep-learning model for the one thing only a model can do well (recognize woody vegetation from imagery), wrapped by a GIS post-processing step that applies the deterministic spatial rules utilities actually care about (distance to lines, terrain steepness). This mirrors how a production vegetation-management workflow is architected.

## Why this project

Utilities like PG&E, SCE, and SDG&E manage vegetation encroachment along thousands of miles of distribution lines. Remote-sensing-driven vegetation mapping is a core input to wildfire-risk models and crew prioritization. A useful workflow needs:

- reliable multi-band satellite ingestion
- a segmentation model that generalizes across landscapes
- a GIS layer that turns "is this woody?" into "is this worth a truck roll?"
- cost-aware model selection (foundation models vs. lean baselines)

This repo benchmarks the segmentation half on the same task and same splits, then plugs the best model into a downstream risk raster.

## Architecture — hybrid ML + GIS

```
                  ┌────────────────────────────┐
                  │ Sentinel-2 L2A (B02–B08)   │
                  │ Dynamic World land cover   │
                  └─────────────┬──────────────┘
                                │
       ┌────────────────────────┴────────────────────────┐
       │                                                  │
       ▼                                                  ▼
 features.tif (7 ch)                           Standalone GIS layers
 [B02, B03, B04, B08, NDVI, sin_doy, cos_doy]  - distance.tif  (meters)
       │                                       - terrain.tif   (elev, slope, aspect×2)
       │                                       - dem.tif       (raw DEM)
       ▼                                                  │
  U-Net (SMP, ResNet50 encoder)                           │
       │                                                  │
       ▼                                                  │
  woody_prob.tif                                          │
  P(trees ∪ shrub/scrub)                                  │
       │                                                  │
       └──────────────────────┬───────────────────────────┘
                              ▼
                 scripts/build_risk_layer.py
                 risk = woody_prob
                      · exp(-distance_m / λ)
                      · clip(slope_deg / slope_scale, 0, 1)
                              │
                              ▼
                         risk.tif  ∈ [0, 1]
```

Two things that fall out of this design:

- The model never sees `distance.tif`, so there's no leakage between the "what is this pixel?" question and the "how far is it from a line?" rule.
- Every knob used by the GIS step (λ, slope scale) is a named, tunable constant with a domain rationale — not a model hyperparameter baked into opaque weights.

## Results

Preferred model:

- task: binary woody / non-woody segmentation
- training AOIs: `bay_area`, `napa`, `sonoma`
- external stress-test AOIs: `lake_county_south`, `sac_wui`, `sac_urban`
- config: `configs/unet_woody_baynapasonoma.yaml`
- best checkpoint: `checkpoints/unet_woody_baynapasonoma/best-08-0.788.ckpt`

Headline numbers:

- internal pooled test `mIoU`: `0.7333`
- internal pooled test woody IoU: `0.8208`
- Sacramento WUI agreement / IoU vs Dynamic World: `0.8297` / `0.6640`
- Sacramento WUI skill above trivial always-woody IoU baseline: `+0.2800`
- Sacramento urban agreement / IoU vs Dynamic World: `0.9590` / `0.4227`
- Lake County South agreement / IoU vs Dynamic World: `0.9259` / `0.9203`
- Lake County South woody coverage match: `89.31%` vs `89.32%`

Interpretation:

- Broader geographic training improved transfer far more than the archived class-remapping and smaller-pool experiments.
- The strongest nontrivial external result is Sacramento WUI, not Lake County South, because Lake County is heavily woody and therefore inflates raw IoU.
- Lake County South remains useful as a calibration check: the model does not collapse in a highly woody unseen AOI and closely matches Dynamic World coverage.
- External behavior is AOI-dependent: strongest in woody/WUI terrain, weaker in sparse urban vegetation.
- This remains a model-vs-Dynamic-World result, not an independent ground-truth result.

More detailed metrics are recorded in `results.md`. Older ablations and intermediate artifacts were intentionally omitted from the public portfolio version.

### DW-direct baseline

Dynamic World already provides tree and shrub/scrub classes that can be combined into a woody-vegetation proxy. A fair portfolio has to answer the question: *does training a model actually buy anything over piping those DW labels straight into `build_risk_layer.py`?*

We answer it honestly with a no-model baseline (`scripts/dw_direct_woody.py`) that binarizes DW labels → `woody_prob_dw.tif`, feeds that through the same GIS risk step, and compares the two pipelines with `scripts/compare_woody.py`.

**What this run actually demonstrates.**

- Both pipelines run end-to-end on the same AOIs and produce comparable rasters with measurable agreement, IoU, coverage, and edge-disagreement statistics.
- The architecture cleanly separates the "what is this pixel?" question (model) from the "how should we act on it?" question (deterministic GIS rules) — which is the property a utility-side workflow needs.
- A continuous model output (vs. DW's categorical hard label) gives downstream tooling a tunable triage threshold, although calibration would require independent validation data.

**What this run does *not* claim.**

- It does not claim to beat Dynamic World on Dynamic World's native task. DW was trained on a far larger expert-labeled corpus with a deeper input stack (10+ S2 bands including red-edge and SWIR). Because the current model is supervised on DW-derived weak labels, it should not be interpreted as outperforming DW as a land-cover classifier without independent validation labels.
- It does not transfer to other sensors. The model expects exactly 4 S2 bands (B02, B03, B04, B08) plus NDVI and seasonal-phase channels. Running it on Landsat, NAIP, or commercial feeds would require retraining from scratch.
- It does not fill DW's cloud gaps. S2 cloud gaps are DW's gaps; the model takes S2 imagery as input and has no separate compositing step.
- It does not distinguish species, age, or fuel state from spectral signal alone — those are below what 10 m single-date S2 carries.

**Why train at all, then.**

The defensible reason is not "build a better land-cover classifier." It is that a wildfire-mitigation taxonomy is not Dynamic World's:

- Ignition risk depends on species, fuel state (dry chaparral vs irrigated landscaping), structure (tall narrow crown vs spreading canopy vs short shrub), and recent burn history. DW exposes none of these directly.
- Many of those distinctions are not recoverable from single-date 10 m Sentinel-2 imagery alone. Dry chaparral and irrigated landscaping look similar at one time point but separate cleanly in multi-temporal SWIR (NDMI). Tall narrow eucalyptus stands and oak woodland look spectrally similar at 10 m but are structurally distinct in LiDAR canopy height.
- DW's frozen weights cannot ingest those signals. A trainable pipeline can.

The repo is structured so the input stack and the label source are swappable components. The current run uses DW labels with a 4-band S2 + NDVI + seasonal-phase input as a baseline. The natural extensions, in roughly increasing payoff order:

1. **Multi-temporal S2** (12-month NDVI / NDMI / NBR stacks) — separates dry chaparral from irrigated cover, captures phenology, near-zero additional infrastructure cost.
2. **LiDAR canopy height** — adds vegetation structure, the signal needed for crown-shape and height-class distinctions. CAL FIRE and utility-corridor LiDAR exist for much of California.
3. **CALVEG and LANDFIRE FBFM40 as additional input bands** — California-specific priors on vegetation type and fire-behavior fuel models that already encode utility-relevant distinctions.
4. **Crew-validated inspection labels in place of DW** — once the input stack carries the right signal, a few hundred labeled tiles meaningfully outperform DW on the slice of the problem that matters: vegetation immediately adjacent to distribution lines.

If the model doesn't beat DW-direct on the current weak-label task, that is the expected result for a student trained on a stronger teacher's outputs. The portfolio value here is the architecture and the methodology, not the headline IoU.

## Repo structure

```
wildfire-veg-seg/
├── configs/                 # YAML training configs
│   └── unet_woody_baynapasonoma.yaml   # preferred 3-AOI binary config
├── data/
│   ├── raw/                 # downloaded tiles + features.tif + GIS layers (gitignored)
│   └── processed/           # tiled chips + masks (gitignored)
├── scripts/
│   ├── download_data.py          # Sentinel-2 + Dynamic World → s2.tif, labels.tif
│   ├── build_geo_features.py     # features.tif (7ch) + distance.tif + terrain.tif
│   ├── make_chips.py             # tile rasters → .npy chips
│   ├── pool_chip_dirs.py         # merge multiple AOI chip folders
│   ├── train.py                  # Lightning training
│   ├── predict.py                # trained model → woody_prob.tif
│   ├── dw_direct_woody.py        # DW labels → woody_prob_dw.tif (no-model baseline)
│   ├── compare_woody.py          # A vs B agreement / IoU / edge analysis
│   └── build_risk_layer.py       # woody_prob + distance + slope → risk.tif
├── src/
│   ├── data/                # download, geo_features, dataset, tiling, transforms
│   ├── models/              # U-Net (Prithvi / Clay backbones planned, see "Model choices")
│   └── training/            # lightning modules, losses, metrics
├── checkpoints/             # optional local model weights (gitignored)
├── outputs/                 # local predictions / figures (gitignored)
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1b. One-time Earth Engine auth (needed for Dynamic World labels)
earthengine authenticate
export EE_PROJECT=<your-gcp-project-with-ee-enabled>

# 2. Download and process the three training AOIs
python scripts/download_data.py --aoi bay_area --start 2024-06-01 --end 2024-09-30 --ee-project $EE_PROJECT
python scripts/download_data.py --aoi napa --start 2024-06-01 --end 2024-09-30 --ee-project $EE_PROJECT
python scripts/download_data.py --aoi sonoma --start 2024-06-01 --end 2024-09-30 --ee-project $EE_PROJECT

python scripts/build_geo_features.py --raw-dir data/raw/bay_area --date 2024-07-15
python scripts/build_geo_features.py --raw-dir data/raw/napa --date 2024-07-15
python scripts/build_geo_features.py --raw-dir data/raw/sonoma --date 2024-07-15

python scripts/make_chips.py --raw-dir data/raw/bay_area --out-dir data/processed/bay_area_woody --image-name features.tif --min-valid-frac 0.5
python scripts/make_chips.py --raw-dir data/raw/napa --out-dir data/processed/napa_woody --image-name features.tif --min-valid-frac 0.5
python scripts/make_chips.py --raw-dir data/raw/sonoma --out-dir data/processed/sonoma_woody --image-name features.tif --min-valid-frac 0.5

python scripts/pool_chip_dirs.py \
  --source bay_area data/processed/bay_area_woody \
  --source napa data/processed/napa_woody \
  --source sonoma data/processed/sonoma_woody \
  --out-dir data/processed/bay_napa_sonoma_woody

# 3. Train the preferred binary model
python scripts/train.py --config configs/unet_woody_baynapasonoma.yaml

# 4. Download and process the held-out fourth AOI
python scripts/download_data.py --aoi lake_county_south --start 2024-06-01 --end 2024-09-30 --ee-project $EE_PROJECT
python scripts/build_geo_features.py --raw-dir data/raw/lake_county_south --date 2024-07-15

# 5. Run inference on the held-out AOI
python scripts/predict.py --config configs/unet_woody_baynapasonoma.yaml \
                          --ckpt checkpoints/unet_woody_baynapasonoma/best-08-0.788.ckpt \
                          --features data/raw/lake_county_south/features.tif \
                          --out outputs/lake_county_south/woody_prob.tif

# 6. Compare against the Dynamic World baseline
python scripts/dw_direct_woody.py \
    --labels data/raw/lake_county_south/labels.tif \
    --out outputs/lake_county_south/woody_prob_dw.tif

python scripts/compare_woody.py \
    --a outputs/lake_county_south/woody_prob.tif \
    --b outputs/lake_county_south/woody_prob_dw.tif

# 7. (Optional) Combine with GIS layers → risk.tif
python scripts/build_risk_layer.py \
    --woody-prob outputs/lake_county_south/woody_prob.tif \
    --distance   data/raw/lake_county_south/distance.tif \
    --terrain    data/raw/lake_county_south/terrain.tif \
    --out        outputs/lake_county_south/risk.tif
```

Archived ablations and intermediate experiments were intentionally omitted from the public portfolio version.

## Data sources

- **Sentinel-2 L2A** (10 m, multispectral) via [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — no auth required.
- **Dynamic World v1** labels (Google) via [Google Earth Engine](https://earthengine.google.com/) (`GOOGLE/DYNAMICWORLD/V1`) — near-real-time 10 m land cover. Used as weak supervision for the segmentation task. Woody = `{1: trees, 5: shrub_scrub}`. Requires a one-time `earthengine authenticate` and a Google Cloud project with the Earth Engine API enabled (pass it via `--ee-project` or `EE_PROJECT`).
- **Copernicus DEM GLO-30** for elevation / slope / aspect.
- **OpenStreetMap** `power=line` features for distance-to-infrastructure.
- *(Optional extension)* **USGS 3DEP LiDAR** for canopy height / higher-resolution slope.

## What the model learns vs. what the GIS step enforces

| Question | Answered by | Why |
|---|---|---|
| Is this pixel woody vegetation? | U-Net on features.tif | Non-linear visual pattern recognition (texture, NDVI, seasonality) — cheap for a CNN, hard to hand-write. |
| Is this pixel near a power line? | `distance.tif` in meters | Deterministic geometry. No model needed. Baking it into the input would leak with the downstream "within X m" rule. |
| Is the slope steep enough to matter? | `terrain.tif` slope band | Same. A literal threshold is more defensible to a utility reviewer than a learned weight. |
| How should these three combine? | `build_risk_layer.py` formula | A simple product with named, tunable constants (`λ`, `slope_scale_deg`) so domain experts can audit / adjust without retraining. |

## How location information enters the model

"Geospatial data" isn't just pixels — it's pixels *anchored to the Earth*. The headline 7-channel model is deliberately spectral + seasonal only, but we support an ablation with explicit geographic positional encoding. When enabled it injects sinusoidal Fourier features of (lat, lon) as extra input channels.

**1. Seasonal encoding** (always on, 2 channels)

`sin(2π · DOY/365)` and `cos(2π · DOY/365)` — seasonal phase as a pair so the January/December boundary is continuous. Same July grass in California is dry and flammable; February grass is green. Without a seasonal signal the model has to guess.

**2. Physical / terrain covariates** (ablation only, 4 channels)

DEM-derived elevation (z-scored), slope (normalized), and aspect split into `sin(aspect)` and `cos(aspect)` so the angular wrap-around at 0°/360° disappears. In the headline config these live in `terrain.tif` and are consumed only by `build_risk_layer.py`.

**3. Utility-relevant spatial covariate** (ablation only, 1 channel)

Distance to the nearest OpenStreetMap `power=line`, in meters. Computed by rasterizing the line geometry onto the reference grid, then running a Euclidean distance transform. In the headline config this lives in `distance.tif` and is consumed only by `build_risk_layer.py`.

**4. Optional — explicit geographic positional encoding** (ablation only, 4 × `num_frequencies` channels)

Per-pixel sinusoidal Fourier encoding of (lat, lon):

```
for k in [0 .. num_frequencies-1]:
    sin(π · 2^k · lat_norm), cos(π · 2^k · lat_norm),
    sin(π · 2^k · lon_norm), cos(π · 2^k · lon_norm)
```

**Why not feed raw (lat, lon) directly?**

- *Spectral bias.* Neural networks are biased toward low-frequency functions of their inputs (NeRF, Tancik et al. 2020). Two adjacent pixels at 37.700 vs 37.701 produce almost-identical inputs; without a Fourier expansion the model can't learn fine-grained spatial distinctions.
- *Scale.* Raw coordinates like `(-122.4, 37.7)` are numerically different from reflectance values in [0, 1] and break input normalization.
- *Sphere geometry.* (0, 180) and (0, −180) are the same place; raw coords create artificial discontinuities the network has to learn around.
- *Generalization.* Raw coords invite the model to *memorize* training locations.

Sinusoidal Fourier features are still *absolute* positional encoding — they just encode absolute position in a form the network can exploit. This is different from the "absolute PE doesn't generalize" critique from the Transformer literature, which is about learned embeddings of discrete token positions, not continuous geographic coordinates.

**Why not SatCLIP / GeoCLIP?**

Those are a strictly better default when the hosted model weights are accessible — they encode learned geographic priors (ecoregion, climate, land use) from image–location contrastive pretraining. We kept this repo framework-light and used hand-crafted Fourier features so the pipeline has zero external embedding dependency. Swapping `latlon_fourier_channels` for a frozen SatCLIP encoder is a drop-in upgrade.

**Foundation models already do this.** Prithvi, Clay, and related geospatial foundation models encode spatial and temporal context during pretraining. The Fourier-PE ablation is both a useful standalone model and a diagnostic: it tells us how much of a foundation-model's lift comes from "knowing where/when it is" vs. deeper pretrained representations.

## Model choices — why

**Currently shipped in this repo:**

- **U-Net (SMP)**: simplest credible baseline. ResNet50 encoder pretrained on ImageNet. Well-understood, fast, robust with limited data. Handles 3–7 bands cleanly. This is the model used for all results reported in `results.md`.

**Planned next (not yet in this repo):**

- **Prithvi-EO-2.0 (IBM/NASA, ViT)**: pretrained on Harmonized Landsat–Sentinel data. Designed for S2-style inputs. Open weights on HuggingFace. Expected to beat U-Net on multi-band input and low-label regimes.
- **Clay** (alternative foundation model) — multi-sensor, more flexible if LiDAR is added later.

**Considered and rejected:**

- ViT from scratch (no chance with this data scale).
- SAM (great for zero-shot coarse objects but not fine vegetation classes without fine-tuning).

## License

MIT.
