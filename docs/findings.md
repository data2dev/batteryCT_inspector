# Findings and status (as of 2026-11-22)

This is an honest account of what the project has and has not shown, and what it
would take to go further. Numbers come from the committed run outputs in
`results/runs/` (and the PatchCore run in PR #19).

## TL;DR

- **The pipeline works** and is reproducible: Figshare → private Kaggle dataset →
  offline Kaggle runs pinned to a commit → results PRs. Two PCA runs from different
  commits produced identical scores.
- **The PatchCore method works on planted anomalies**: on the full Li-ion 18650
  cohort (406 cells), it flags 85% and localizes 95% of 60 synthetic anomalies, and
  reliably finds anything ≳ 0.15 mm.
- **We found no convincing real defect.** The cohort is very uniform. The cells that
  stand out most are scans that are missing from the dataset's own inventory (likely
  a different cell or scan setup), not demonstrated defects.
- **The original goal, a defect detector validated on real defects, is not reachable
  with this data and this team** as things stand: there are no labels, neither team
  member can tell a defect from normal variation by eye, and no expert is involved.
- What exists is a **tested, open screening tool with a measured sensitivity**, plus
  a documented negative result on one commercial cohort.

## What was built

| Component | State |
|---|---|
| Ingest: range requests into the 91 GB Figshare zip, 7 views/cell, CRC + sha256 | Done; li18650: 406 scans, 2,842 images, 6.0 GB |
| Offline Kaggle runs: wheel built at the commit and embedded; results PR | Done |
| PCA residual baseline (view-balanced), review pack | Done |
| PatchCore for unlabeled cohorts: cross-fitting, k-th NN, trimming, position-free banks | Done (GPU and CPU configs) |
| Planted-anomaly benchmark (particles, voids, gaps; 6–36 px) | Done |
| Hotspot review pack: crop next to nearest match in another cell | Done |
| Electrode-edge measurement (overhang, alignment) | **Not started** |
| Descriptive labels, expert review | **Not started** |

## Results

### Runs

| Run | Method | Cells | Runtime | Notes |
|---|---|---|---|---|
| `20261120-0330_li18650-pca_71ff449` | PCA | 406 | 14 min CPU | before view balancing |
| `20261120-0507_li18650-pca_75a7982` | PCA, balanced | 406 | 16 min CPU | |
| `20261122-0804_li18650-pca_6d85de6` | PCA, balanced | 406 | 16 min CPU | identical scores to 0507 (reproducibility) |
| `20261122-0125_li18650-patchcore-cpu_6d85de6` (PR #19) | PatchCore, CPU | 406 | 5.95 h CPU | 5% random bank, 1 pass, 60 planted |

### PatchCore: planted anomalies (synthetic, n = 60)

| | Flagged (> 95th pct of real cells) | Located |
|---|---|---|
| All | 85% | 95% |
| Particle / void / gap | 95% / 80% / 80% | 100% / 95% / 90% |
| ≥ 24 px (≥ ~0.35 mm) | 100% | 100% |
| 16 px (~0.2 mm) | 92% | 100% |
| 10 px (~0.15 mm) | 88% | 100% |
| 6 px (~0.1 mm) | 46% | 77% |

Pixel sizes assume ~14 µm/px, estimated from the cell diameter. The dataset does not
state the voxel size.

**Caveat:** these are synthetic anomalies drawn into real images (bright disks, dark
ellipses, thin dark lines). Real defects may look different and be harder to find.
This is a sensitivity estimate, not proof that real defects would be found.

### Real cells

- **PatchCore** scores are tightly bunched: median 1.0 by construction, 99th
  percentile 1.18, max 1.40. One cell is clearly separate (1958, score 1.40; the
  next is 1.25).
- **PCA** finds a group of 7 cells far from the rest (scores 30–85, vs ≤ 11 for all
  others). Their differences are spread over whole images, not local spots.
- **Agreement:** both methods rank **1958** #1 and **1826** #2. Otherwise the
  rankings are almost unrelated (Spearman ρ = 0.17): PCA reacts to whole-cell
  patterns, PatchCore to local spots.
- **Scans missing from `Scan_inventory.csv`** (6 of 406) rank #1, #2 and #7 in
  PatchCore and 4 of them rank in the PCA top 15 (Mann-Whitney p = 0.006, first PCA
  run). In the sodium-ion cohort (local analysis, not committed) 4 of 5 such scans
  have a visibly different winding (more, thinner layers), i.e. a different design.
- **Serial-number lot codes** (e.g. KLAQ31, LLAQ33) differ in PCA score
  (Kruskal-Wallis p = 0.004, first PCA run): some flagged variation is lot-to-lot.

### What the top PatchCore hotspots look like (descriptive only)

| Rank | Scan | Observation | Reading |
|---|---|---|---|
| 1 | 1958 (not in inventory) | In one cross-section, the bright outer ring (can wall) is missing or faint where layers meet the background; matches elsewhere have a bright ring. Also lower contrast and a differently shaped top cap in other views. | Likely a different cell or scan setup; not shown to be a defect |
| 3 | 1883 | At the bottom corner of a lengthwise slice, a bright line has an S-shaped kink where its closest match is straight. | **Candidate for expert review** (possible deformation at the cell bottom) |
| 4 | 2008 | Hotspot inside the empty centre hole; the match looks similar. | **False alarm** (empty region) |
| 5 | 2087 | Near the bottom of the centre channel, a faint line curves where the match's is straight. | Mild; unknown |
| — | several (smoke run) | End of the outermost electrode strip under the can wall (each cell has one, at a random angle). | Rare-but-normal structure; false alarm |

## Why we stop short of "defect detection"

1. **No ground truth.** The dataset has no defect labels.
2. **No expertise to make labels.** Labeling by eye would be guessing, so the project
   deliberately recorded only descriptions, not "defect" calls.
3. **The cohort is probably clean.** It is one model (INR18650/33V) from one batch.
   Lumafield's CT study of 1,054 commercial 18650 cells found its clearest defect
   (cathode overhang, 33 cells) only in low-cost or counterfeit brands.
4. **Synthetic recall is not real recall.** Planted anomalies show the method *can* see
   ~0.15 mm local changes; they say nothing about which real defects exist or how they
   look.

## Known limitations of the current methods

- **Resolution:** images are scaled to ¼ before feature extraction, so features cover
  ~0.45 mm. ~0.1 mm anomalies are found only about half the time.
- **CPU budget:** without a Kaggle GPU (requires phone verification), the bank is a 5%
  random sample and there is no trimming pass. On 16 cells this cost about one
  planted void out of four and made the ranking noisier.
- **Empty regions** (background, centre hole) can produce false alarms; they are not
  masked yet.
- **Rare-but-normal structures** (electrode-strip ends, tabs) score high when the bank
  covers them thinly.
- **Transductive:** models are fitted on the same cohort they score. That is
  intentional for screening, but means no held-out evaluation.
- **Views:** only 7 of ~174 images per cell are used (4 lengthwise angles, 3 heights).
  Anything between those slices is invisible.

## What it would take to move forward

Ordered from cheapest to most effective.

### To make the current results more trustworthy (days)
- **Mask empty regions** (background, centre hole) before scoring.
- **Descriptive labels** for the top ~30 hotspots in `labels/` (what is seen, not
  whether it is a defect), to measure the real false-alarm rate per 100 cells.
- **Confirm the voxel size** with Glimpse (currently estimated at ~14 µm/px), so sizes
  in mm are exact.
- **Ask Glimpse what the 6 inventory-missing scans are** (rescans? another product?).

### To say something about real defects (weeks, needs people or data)
- **A domain expert** (battery researcher or CT inspection engineer) to review the
  top 20–30 cells for about an hour. This is the single highest-value step.
- **Electrode-edge measurement:** anode overhang, cathode overhang and edge
  alignment at full resolution in the lengthwise slices. These have published reference
  values (anode should extend ~0.5 mm past the cathode; cathode past anode is a
  defect), so "defect" becomes a measured spec violation rather than a judgment.
- **Data with known or likely defects:** cells from low-cost or counterfeit brands,
  deliberately damaged or aged cells, or a dataset with expert annotations. Without
  some cells that are *known* to be defective, no detector can be validated.

### To push the method itself (compute)
- **A GPU** (Kaggle with phone verification, or another free GPU) would allow the full
  bank, the trimming pass and a finer `scale` (0.5) aimed at ~0.1 mm anomalies.
- **More views per cell** (more heights and angles) to reduce what falls between
  slices.

## Options for the project

1. **Wrap up as a tool + benchmark:** finish docs, publish the software with the
   planted-anomaly evaluation and this negative result (e.g. as research software).
2. **Pivot to measurement:** electrode-overhang and alignment statistics across 406
   cells as a manufacturing-variation study.
3. **Find better data**, then return to defect detection.
4. **Pause** here. Everything is reproducible from this repository.
