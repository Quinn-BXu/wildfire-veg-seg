# Results

This file records the current preferred pipeline and the headline result to report from this repo.

## Preferred Pipeline

- Training AOIs: `bay_area`, `napa`, `sonoma`
- Task: binary woody segmentation
  - `0 = non-woody`
  - `1 = woody`
- Preferred config: `configs/unet_woody_baynapasonoma.yaml`
- Preferred checkpoint: `checkpoints/unet_woody_baynapasonoma/best-08-0.788.ckpt`
- External held-out AOIs: `lake_county_south`, `sac_wui`, `sac_urban`

## Internal Pooled Test

Metrics from the pooled Bay Area + Napa + Sonoma binary model:

- `test/iou_class_0 = 0.6459`
- `test/iou_class_1 = 0.8208`
- `test/miou = 0.7333`
- `test/loss = 0.2228`

Interpretation:

- The pooled binary model is stable on an internal held-out split.
- Woody-class performance is strong enough to justify external AOI testing.

## External Transfer: Lake County South

Computed with `scripts/compare_woody.py` against the Dynamic World woody baseline on a fully unseen fourth AOI.

- Model woody coverage: `0.8931`
- Dynamic World woody coverage: `0.8932`
- Model mean woody probability: `0.8536`
- Dynamic World mean woody probability: `0.8932`
- Agreement: `0.9259`
- IoU (model vs Dynamic World): `0.9203`
- Edge disagreement fraction: `0.2257`

Interpretation:

- The 3-AOI binary model transfers to a fully unseen fourth AOI without collapsing.
- The model and Dynamic World have nearly identical woody coverage in Lake County South.

Important caveat:

- This is still a model-vs-Dynamic-World agreement result, not model-vs-independent-ground-truth.
- The safe claim is that the model generalizes well in reproducing the Dynamic World woody mask on a new AOI.
- Lake County South is highly woody (`89.32%` Dynamic World coverage), so raw IoU is inflated by class balance and should not be the main headline metric.

## Additional External Stress Tests

These two Sacramento-area AOIs were evaluated with the same preferred 3-AOI binary model to check behavior in lower-coverage urban fabric and mixed WUI terrain.

### Sacramento Urban

- Model woody coverage: `0.0614`
- Dynamic World woody coverage: `0.0398`
- Model mean woody probability: `0.0714`
- Dynamic World mean woody probability: `0.0398`
- Agreement: `0.9590`
- IoU (model vs Dynamic World): `0.4227`
- Edge disagreement fraction: `0.1995`

Interpretation:

- Raw agreement and raw IoU are lower than in heavily woody AOIs because the sparse-coverage setting is inherently harder.
- Measured as skill above the trivial always-woody baseline, this is the largest IoU delta across the three external AOIs, which suggests the model is finding real sparse vegetation rather than just matching coverage priors.
- That said, sparse urban coverage is likely the noisiest reference regime: Dynamic World is likely least reliable at very low woody fraction (individual street trees, backyard canopy, urban lawn-vs-shrub edges). So this counts as supporting anti-trivial evidence, not the main operational headline — that role goes to `sac_wui`.
- The model is somewhat more liberal than Dynamic World, but it does not collapse or predict woody everywhere.

### Sacramento WUI

- Model woody coverage: `0.4593`
- Dynamic World woody coverage: `0.3840`
- Model mean woody probability: `0.4394`
- Dynamic World mean woody probability: `0.3840`
- Agreement: `0.8297`
- IoU (model vs Dynamic World): `0.6640`
- Edge disagreement fraction: `0.1588`

Interpretation:

- On balanced coverage (`~38%` woody), the model reaches `0.6640` IoU against Dynamic World, which is the strongest operationally relevant external result in the repo.
- The model remains somewhat more liberal than Dynamic World, but the disagreement is moderate and spatially plausible for an unseen AOI.

### Cross-AOI Summary

- `sac_wui`: strongest operationally relevant external result in a mixed WUI landscape
- `sac_urban`: strongest sparse-regime anti-trivial result, but less central to the wildfire use case
- `lake_county_south`: high-coverage calibration check in a heavily woody landscape

This pattern is coherent:

- the model remains useful across very different coverage regimes
- the most relevant external skill signal is in WUI terrain
- the high-coverage woodland case works mainly as a sanity check, not as the headline metric

## Trivial-Baseline Interpretation

For binary woody IoU against Dynamic World, a trivial `always predict woody` baseline equals the Dynamic World woody coverage.

| AOI | DW woody coverage | Model IoU vs DW | Trivial always-woody IoU | Skill above trivial baseline |
|---|---:|---:|---:|---:|
| `lake_county_south` | `0.8932` | `0.9203` | `0.8932` | `+0.0271` |
| `sac_wui` | `0.3840` | `0.6640` | `0.3840` | `+0.2800` |
| `sac_urban` | `0.0398` | `0.4227` | `0.0398` | `+0.3829` |

Interpretation:

- `sac_wui` is the strongest external result to headline because it shows substantial skill above a trivial baseline in the regime most relevant to wildfire operations.
- `lake_county_south` still matters, but mainly as evidence that the model does not collapse in a heavily woody unseen AOI.
- `sac_urban` is more informative through woody IoU than through agreement, because sparse urban vegetation makes agreement near-trivial.

For overall agreement, the trivial baseline is `max(coverage, 1 - coverage)`.

| AOI | Model agreement | Trivial agreement | Skill above trivial baseline |
|---|---:|---:|---:|
| `lake_county_south` | `0.9259` | `0.8932` | `+0.0327` |
| `sac_wui` | `0.8297` | `0.6160` | `+0.2137` |
| `sac_urban` | `0.9590` | `0.9602` | `-0.0012` |

Interpretation:

- Agreement should not be used to headline `sac_urban`.
- The urban AOI is still useful, but its meaningful metric is woody IoU against a very low-coverage baseline, not raw agreement.

## Cross-AOI Calibration Note

Absolute woody coverage by AOI:

- `lake_county_south`: `0.8931` vs `0.8932` → effectively matched
- `sac_wui`: `0.4593` vs `0.3840` → `+0.0753` absolute overprediction
- `sac_urban`: `0.0614` vs `0.0398` → `+0.0216` absolute overprediction

Interpretation:

- The model has a consistent positive woody bias across AOIs.
- In a screening workflow this may be acceptable if recall is prioritized, but for production use the threshold should be calibrated explicitly.

## Preferred Narrative

The main engineering result is:

> A binary U-Net trained on Bay Area + Napa + Sonoma transfers across unseen AOIs spanning sparse urban vegetation, mixed WUI terrain, and heavily woody woodland. The strongest nontrivial external result is Sacramento WUI, where the model reaches `0.6640` IoU against the Dynamic World woody baseline, or `+0.2800` above the trivial always-woody baseline, while Lake County South shows that the model remains calibrated and does not collapse in a high-coverage woody landscape.

## Archived Experiments

Older experiments, intermediate checkpoints, and non-winning outputs were intentionally omitted from the public portfolio version. These included:

- Bay Area-only experiments
- Bay Area + Napa pooled experiments
- the 4-class ablation
- older preview maps and diagnostic figures
