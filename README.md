# BatteryCT-Inspector

[![CI](https://github.com/data2dev/batteryct_inspector/actions/workflows/ci.yml/badge.svg)](https://github.com/data2dev/batteryct_inspector/actions/workflows/ci.yml)

Open, reproducible anomaly screening for X-ray CT cross-sections of same-design battery cells.

> **Status: pre-alpha.** The package currently contains I/O, preprocessing and a synthetic
> test-data generator. Anomaly models and real-data results are not published yet.

## What it is for

Engineers inspecting a batch of same-design cells (for example 18650 or 21700 cylindrical cells)
want to know which scans look different from the rest and deserve a closer look. BatteryCT-Inspector
aims to rank cells by how much they differ from the cohort, show *where* they differ, and report false
alarms honestly, including those caused by scan artifacts such as beam hardening and metal streaks.

**An anomaly score means "differs from the reference cohort." It does not mean a cell is unsafe or
defective unless an expert has reviewed it.**

## Install (development)

```bash
git clone https://github.com/data2dev/batteryct_inspector.git
cd batteryct_inspector
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
batteryct --version
# Write a small SYNTHETIC cohort (procedurally generated, not real CT data)
batteryct synth ./synth_demo --n-normal 20 --n-per-defect 2
```

```python
from batteryct.io import load_image
from batteryct.preprocess import normalize_percentile, foreground_centroid, center_crop

img = normalize_percentile(load_image("slice.png"))
crop = center_crop(img, 512, center=foreground_centroid(img))
```

## Data

The study uses the public battery CT dataset released by Glimpse:

> Condon, A. et al. *A dataset of over one thousand computed tomography scans of battery cells.*
> Data in Brief 55 (2024) 110614. https://doi.org/10.1016/j.dib.2024.110614
> Dataset: https://doi.org/10.25452/figshare.plus.25330501

That dataset is licensed **CC BY-NC-SA 4.0** (non-commercial, share-alike). This repository does
not contain the original images. Anything here derived from them is under the same license; see
[`DATA_LICENSES.md`](DATA_LICENSES.md).

Tests and CI use **synthetic** images only (`batteryct.synthetic`), which are not derived from the
dataset.

## Repository layout

```
src/batteryct/   library code (Apache-2.0)
tests/           unit tests on synthetic fixtures
configs/         experiment configs
results/         metrics, splits, small figures (CC BY-NC-SA 4.0)
labels/          manual review labels (CC BY-NC-SA 4.0)
docs/            documentation
```

## License

- Code: [Apache-2.0](LICENSE)
- Data-derived content (`results/`, `labels/`, published images and model weights):
  CC BY-NC-SA 4.0, see [`DATA_LICENSES.md`](DATA_LICENSES.md)

## Citation

See [`CITATION.cff`](CITATION.cff). If you use results derived from the dataset, also cite the
Glimpse data paper above.
