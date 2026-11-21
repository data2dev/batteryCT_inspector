# PatchCore on an unlabeled cohort

Standard PatchCore assumes the training images are all normal. Here nobody knows
which cells are normal, so the cohort is scored against *itself*, with three guards
against a rare anomaly being "explained away" by the memory bank.

## How a cell is scored

1. **Features.** Each prepared view (radial views cropped around the cell) is scaled
   by `scale` (default 0.25), passed through an ImageNet-pretrained WideResNet-50-2,
   and layer2 + layer3 activations are averaged over 3×3 neighbourhoods,
   concatenated (1536-d) and projected to `proj_dim` (256) with a fixed random
   matrix. One feature = one **patch** on an 8-pixel grid of the scaled image
   (about 0.45 mm of cell at the default scale).
2. **Banks per view type.** Axial patches are compared with axial patches, radial
   with radial. Position is ignored, so a cell rotated in the scanner is not
   penalized for its tabs being elsewhere.
3. **Cross-fitting (`folds`).** Cells are split into folds. Each fold is scored
   against a bank built only from the *other* folds, so no cell can match itself.
4. **k-th nearest neighbour (`k`).** A patch's score is its distance to the k-th
   closest bank patch. A pattern present in fewer than about k other cells keeps a
   high score; a pattern in k or more other cells starts to look normal.
5. **Trimming (`trim_top_frac`).** After a first pass, the top 2 % of cells are
   removed from every bank and everything is re-scored.
6. **Cell score.** Per view type: the mean of the cell's `top_patches` highest patch
   scores, divided by that view type's median over cells. Overall score: the max
   over view types.

## What it can and cannot find

- **Can:** anything that is rare at the patch level, i.e. one-off or few-off local
  differences (particles, voids, gaps, odd structures), wherever they sit.
- **Cannot:** a defect *type* that recurs in many cells (≥ about `k` others). That is
  what explicit measurements (e.g. anode overhang in mm) are for.
- **Resolution:** anything much smaller than one patch (≈ 0.45 mm at `scale: 0.25`)
  is diluted. The planted-anomaly test reports recall by size.

## Planted anomalies (recall without labels)

`inject` copies of randomly chosen real cells get one synthetic anomaly each in one
view: a bright **particle** (dense debris), a dark **void** (gas pocket) or a thin dark
**gap**, with radius / half-length 6–36 px (≈ 0.1–0.5 mm at ~14 µm/px). Copies are
only ever *queries*: they are scored against the same bank as their source cell's
fold (which excludes the source), so they never affect real cells.

`planted.csv` and `metrics.json` → `planted` report:

| Field | Meaning |
|---|---|
| `cell_detected_p95` / `_p99` | the copy scores above the 95th / 99th percentile of real cells |
| `located` | a patch on or next to the planted spot scores above the 99.9th percentile of real patches |
| `by_type`, `by_size_px` | the same, broken down |

These are synthetic anomalies; recall on real defects can differ.

## Review pack

`review/NN_<scan>.jpg` for the top `hotspots` cells: the view with the hotspot boxed,
a full-resolution crop of it, and the **most similar patch found in any other cell**.
The reviewer's question is only *"is the middle crop genuinely unlike the right
one?"*. No battery expertise is needed to answer it. Whether a difference is a
defect, benign variation or a scan artifact is a separate question (measurement or
expert).

## Running it

0. Kaggle account **phone-verified** (Kaggle only grants GPUs to verified accounts).
1. Once: **Actions → Upload model weights to Kaggle** (creates the private dataset
   `<you>/batteryct-weights`, hash-verified).
2. Smoke test: **Run experiment on Kaggle** → `configs/experiments/li18650_patchcore_smoke.yaml`
   (40 cells).
3. Full: `configs/experiments/li18650_patchcore.yaml` (GPU, about an hour).

Reference: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*
(PatchCore), CVPR 2022. For training on contaminated data see Jiang et al.,
*SoftPatch*, NeurIPS 2022.
