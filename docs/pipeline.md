# Pipeline: ingest → experiment → results

```
Figshare zip ──(range requests, GitHub runner)──► private Kaggle dataset  ──┐
   "Ingest cohort" workflow                        + manifest PR            │
                                                                           ▼
configs/experiments/*.yaml ──"Run experiment on Kaggle"──► Kaggle kernel (pinned SHA)
                                                                           │
            results/runs/<run_id>/  ◄── PR ◄── collect (same job, or every 3 h) ┘
```

## One-time setup

1. Repo secret `KAGGLE_API_TOKEN` (Kaggle → Settings → API → **API Tokens** → Generate). Your username is
   looked up from the token. Legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` (kaggle.json) still work as a fallback.
   Later: `HF_TOKEN` (Hugging Face write token) for publishing.
2. Settings → Actions → General → *Allow GitHub Actions to create and approve pull requests*.
3. Kaggle phone verification is **not** required: runs are offline (see below). It is only needed
   for a config that sets `internet: true`.

## Ingest a cohort

Actions → **Ingest cohort** → choose cohort (e.g. `li18650`). Use `limit_scans: 2` for a dry
run. Creates/updates the private dataset `<you>/batteryct-<cohort>`.

## Run an experiment

Actions → **Run experiment on Kaggle** → config path (e.g.
`configs/experiments/li18650_pca.yaml`).

- Builds the `batteryct` wheel from **the exact commit** of the workflow run and embeds it
  (base64, ~20 kB, sha256-checked) in the `run.py` of a private Kaggle *script* kernel
  `<you>/batteryct-<config name>`. Kaggle installs it offline (numpy, Pillow, PyYAML are
  preinstalled), attaches the cohort dataset, and runs `batteryct experiment`.
- Kernels run with **internet off** by default; a config can set `internet: true` (needs a
  phone-verified Kaggle account).
- Each run gets an id `<UTC date-time>_<config name>_<short sha>`; outputs go to
  `/kaggle/working/<run_id>/`.
- The job waits up to `wait_minutes` (default 330), then copies the small outputs
  (`scores.csv`, `metrics.json`, `run_meta.json`, `config.yaml`, contact sheets, log tail) into
  `results/runs/<run_id>/` and opens a PR.
- Only one Kaggle run at a time (`concurrency: kaggle-runs`).
- Runs longer than the GitHub job are collected by **Collect Kaggle results** (every 3 h).
- `gpu: true` in a config requests a Kaggle GPU; the PCA baseline is CPU-only.

## Run outputs

| File | Content |
|---|---|
| `scores.csv` | cells ranked by score; residual, T², the view driving the score, inventory info |
| `metrics.json` | config, counts, dropped scans, explained variance, score quantiles, runtime |
| `run_meta.json` | run id, git SHA, status, runtime, Python/GPU |
| `top_outliers.png`, `most_typical.png` | contact sheets (CC BY-NC-SA 4.0, derived from Glimpse data) |

A score means "differs from the rest of the cohort", not "defective", until a person has
reviewed the cell.
