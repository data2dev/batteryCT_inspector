# Data licenses

This repository mixes two kinds of content under different licenses.

| Content | Location | License |
|---|---|---|
| Source code, configs, tests, workflows | everything except the folders below | [Apache-2.0](LICENSE) |
| Metrics, splits, manifests, small figures derived from the dataset | [`results/`](results/) | [CC BY-NC-SA 4.0](results/LICENSE) |
| Manual review labels for dataset images | [`labels/`](labels/) | [CC BY-NC-SA 4.0](labels/LICENSE) |
| Processed images, heatmaps, overlays (Hugging Face dataset repo) | external | CC BY-NC-SA 4.0 |
| Model weights fitted on the dataset (Hugging Face model repo) | external | CC BY-NC-SA 4.0 |

## Source dataset

All data-derived content comes from:

> Condon, A., Buscarino, B., Moch, E., Sehnert, W. J., Miles, O., Herring, P. K., & Attia, P. M.
> *A dataset of over one thousand computed tomography scans of battery cells.*
> Data in Brief 55 (2024) 110614. https://doi.org/10.1016/j.dib.2024.110614
>
> Dataset (Glimpse): https://doi.org/10.25452/figshare.plus.25330501
> License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Changes made

Images derived from the dataset may have been **cropped, resampled, intensity-normalized, and
overlaid** with anomaly heatmaps or annotations. They are not the original images. Originals are
available from the Figshare link above; this project does not redistribute them.

## Terms that follow from the source license

- **Attribution:** credit Glimpse and the data paper wherever derived images or results appear.
- **Non-commercial:** derived content may not be used for commercial purposes.
- **Share-alike:** adaptations of derived content must be released under CC BY-NC-SA 4.0.

The Apache-2.0 license on the code does **not** grant any rights to the dataset or to content
derived from it. Commercial use of the data requires a separate license from Glimpse.

This file is a summary, not legal advice. The license texts linked above govern.

## Synthetic data

Images produced by `batteryct.synthetic` (used in tests and CI) are procedurally generated, are
not derived from the dataset, and are covered by the code license.
