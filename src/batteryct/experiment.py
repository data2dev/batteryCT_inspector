"""Experiment runner: score every cell in an ingested cohort.

First method, ``pca_residual``: a classical, label-free baseline.

1. Each selected image is percentile-normalized, cell-centred (radial views),
   and downsampled to a small fixed grid.
2. A cell's images are concatenated into one feature vector.
3. Features are robustly standardized (median / MAD), projected onto the top
   principal components, and each cell gets two scores: the reconstruction
   residual (structure the main modes of variation can't explain) and Hotelling
   T^2 (an extreme position along the main modes). The final score sums both,
   each divided by its cohort median.

The model is fitted on the whole cohort (no labels exist yet), so a score means
"differs from the rest of this cohort", not "defective".
"""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from batteryct import __version__
from batteryct.io import load_image
from batteryct.preprocess import center_crop, foreground_centroid, normalize_percentile

FloatArray = npt.NDArray[np.float64]
METHODS = ("pca_residual",)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    cohort: str
    dataset: str = ""
    method: str = "pca_residual"
    gpu: bool = False
    views: tuple[str, ...] = ("axial", "radial")
    size: int = 64
    n_components: int = 10
    top_k: int = 12
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> ExperimentConfig:
        for key in ("name", "cohort"):
            if not d.get(key):
                raise ValueError(f"experiment config needs '{key}'")
        method = d.get("method", "pca_residual")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        p = dict(d.get("params", {}) or {})
        cfg = cls(
            name=str(d["name"]),
            cohort=str(d["cohort"]),
            dataset=str(d.get("dataset", "")),
            method=method,
            gpu=bool(d.get("gpu", False)),
            views=tuple(p.pop("views", ("axial", "radial"))),
            size=int(p.pop("size", 64)),
            n_components=int(p.pop("n_components", 10)),
            top_k=int(p.pop("top_k", 12)),
            extra=p,
        )
        if cfg.size < 8 or cfg.n_components < 1 or cfg.top_k < 1:
            raise ValueError("size >= 8, n_components >= 1, top_k >= 1 required")
        return cfg


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class ImageRef:
    scan_id: str
    view: str
    position: float
    path: Path


def find_manifest(data_dir: str | Path) -> Path:
    """Locate ``manifest.csv`` at or below ``data_dir`` (Kaggle may nest folders)."""
    root = Path(data_dir)
    if (root / "manifest.csv").is_file():
        return root / "manifest.csv"
    hits = sorted(root.rglob("manifest.csv"), key=lambda p: len(p.parts))
    if not hits:
        raise FileNotFoundError(f"no manifest.csv under {root}")
    return hits[0]


def load_cohort(data_dir: str | Path) -> tuple[list[ImageRef], dict[str, dict[str, str]]]:
    """Read the ingest manifest and resolve every image path.

    Paths are resolved relative to the manifest; if Kaggle unpacked ``images/``
    one level deeper, files are matched by their last two path components.
    Returns image refs and per-scan metadata.
    """
    manifest = find_manifest(data_dir)
    base = manifest.parent
    rows = list(csv.DictReader(manifest.open()))
    index: dict[str, Path] | None = None
    refs: list[ImageRef] = []
    meta: dict[str, dict[str, str]] = {}
    for r in rows:
        p = base / r["file"]
        if not p.is_file():
            if index is None:
                index = {"/".join(q.parts[-2:]): q for q in Path(data_dir).rglob("*.png")}
            key = "/".join(Path(r["file"]).parts[-2:])
            if key not in index:
                raise FileNotFoundError(f"image listed in manifest not found: {r['file']}")
            p = index[key]
        refs.append(ImageRef(r["scan_id"], r["view"], float(r["position"]), p))
        meta.setdefault(
            r["scan_id"],
            {k: r.get(k, "") for k in ("in_inventory", "cell_serial_number", "cell_model_name")},
        )
    return refs, meta


# ----------------------------------------------------------------------- features


def image_features(path: Path, view: str, size: int) -> FloatArray:
    """Downsample one view to a fixed grid. Radial views are cropped around the cell."""
    img = normalize_percentile(load_image(path))
    if view == "radial":
        side = min(img.shape)
        img = center_crop(img, side, center=foreground_centroid(img, threshold=0.15))
        shape = (size, size)
    else:  # axial: long thin slice; fixed aspect so slightly different heights align
        shape = (size, 4 * size)
    small = Image.fromarray((img * 255).astype(np.uint8)).resize(shape, Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.float64).ravel() / 255.0


@dataclass
class FeatureMatrix:
    scan_ids: list[str]
    X: FloatArray
    blocks: list[tuple[str, int, int]]  # (label, start, stop) per image slot
    slot_paths: list[list[Path]]  # per scan, the image path of each slot
    dropped: dict[str, str]


def build_features(refs: list[ImageRef], cfg: ExperimentConfig) -> FeatureMatrix:
    by_scan: dict[str, dict[str, list[ImageRef]]] = defaultdict(lambda: defaultdict(list))
    for r in refs:
        if r.view in cfg.views:
            by_scan[r.scan_id][r.view].append(r)
    # The slot layout (views x count) is the most common one in the cohort.
    layouts: dict[tuple[tuple[str, int], ...], int] = defaultdict(int)
    for views in by_scan.values():
        layouts[tuple((v, len(views.get(v, []))) for v in cfg.views)] += 1
    if not layouts:
        raise ValueError(f"no images for views {cfg.views}")
    layout = max(layouts, key=lambda k: layouts[k])

    scan_ids, rows, paths, dropped = [], [], [], {}
    blocks: list[tuple[str, int, int]] = []
    for sid in sorted(by_scan):
        views = by_scan[sid]
        if tuple((v, len(views.get(v, []))) for v in cfg.views) != layout:
            dropped[sid] = "image count differs from cohort layout"
            continue
        feats, slot_paths, offset, scan_blocks = [], [], 0, []
        for v, _ in layout:
            for i, ref in enumerate(sorted(views[v], key=lambda r: r.position)):
                f = image_features(ref.path, v, cfg.size)
                feats.append(f)
                slot_paths.append(ref.path)
                scan_blocks.append((f"{v}[{i}]", offset, offset + f.size))
                offset += f.size
        scan_ids.append(sid)
        rows.append(np.concatenate(feats))
        paths.append(slot_paths)
        blocks = blocks or scan_blocks
    if len(rows) < 3:
        raise ValueError(f"need at least 3 complete scans, got {len(rows)}")
    return FeatureMatrix(scan_ids, np.vstack(rows), blocks, paths, dropped)


# ------------------------------------------------------------------------ scoring


@dataclass
class Scores:
    score: FloatArray
    residual: FloatArray
    t2: FloatArray
    block_residual: FloatArray  # (n_scans, n_blocks)
    explained_variance_ratio: FloatArray


def robust_standardize(X: FloatArray) -> FloatArray:
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    scale = np.where(mad > 1e-9, mad, X.std(axis=0) + 1e-9)
    return np.asarray((X - med) / scale)


def pca_residual_scores(
    X: FloatArray, n_components: int, blocks: list[tuple[str, int, int]]
) -> Scores:
    Z = robust_standardize(X)
    Z = np.clip(Z, -20.0, 20.0)  # one wild pixel must not dominate the fit
    mu = Z.mean(axis=0)
    Zc = Z - mu
    k = max(1, min(n_components, Zc.shape[0] - 2, Zc.shape[1]))
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    T = Zc @ Vt[:k].T
    recon = T @ Vt[:k]
    R = Zc - recon
    residual = (R**2).sum(axis=1)
    sd = 1.4826 * np.median(np.abs(T - np.median(T, axis=0)), axis=0)
    sd = np.where(sd > 1e-9, sd, T.std(axis=0) + 1e-9)
    t2 = ((T / sd) ** 2).sum(axis=1)
    score = residual / np.median(residual) + t2 / max(float(np.median(t2)), 1e-9)
    block_res = np.stack([(R[:, a:b] ** 2).sum(axis=1) for _, a, b in blocks], axis=1)
    evr = (S**2) / max(float((S**2).sum()), 1e-12)
    return Scores(score, residual, t2, block_res, evr[:k])


# ------------------------------------------------------------------------ outputs


def _contact_sheet(
    items: list[tuple[str, Path]], out: Path, thumb: int = 160, cols: int = 6
) -> None:
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("L", (cols * thumb, rows * (thumb + 14)), 255)
    draw = ImageDraw.Draw(sheet)
    for i, (label, path) in enumerate(items):
        img = normalize_percentile(load_image(path))
        im = Image.fromarray((img * 255).astype(np.uint8))
        im.thumbnail((thumb, thumb))
        x, y = (i % cols) * thumb, (i // cols) * (thumb + 14)
        sheet.paste(im, (x + (thumb - im.width) // 2, y + 14))
        draw.text((x + 2, y + 1), label, fill=0)
    sheet.save(out)


def run_experiment(cfg: ExperimentConfig, data_dir: str | Path, out_dir: str | Path) -> Path:
    """Score all cells and write ``scores.csv``, ``metrics.json`` and contact sheets."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs, meta = load_cohort(data_dir)
    fm = build_features(refs, cfg)
    s = pca_residual_scores(fm.X, cfg.n_components, fm.blocks)

    order = np.argsort(-s.score)
    worst_block = s.block_residual.argmax(axis=1)
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "scan_id",
                "score",
                "residual",
                "t2",
                "driving_view",
                "driving_image",
                "in_inventory",
                "cell_serial_number",
            ]
        )
        for rank, i in enumerate(order, 1):
            sid = fm.scan_ids[i]
            b = int(worst_block[i])
            w.writerow(
                [
                    rank,
                    sid,
                    f"{s.score[i]:.4f}",
                    f"{s.residual[i]:.2f}",
                    f"{s.t2[i]:.3f}",
                    fm.blocks[b][0],
                    fm.slot_paths[i][b].name,
                    meta[sid]["in_inventory"],
                    meta[sid]["cell_serial_number"],
                ]
            )

    k = min(cfg.top_k, len(order))

    def sheet_items(idx: Any) -> list[tuple[str, Path]]:
        return [
            (
                f"#{int(np.where(order == i)[0][0]) + 1} {fm.scan_ids[i]}",
                fm.slot_paths[i][int(worst_block[i])],
            )
            for i in idx
        ]

    _contact_sheet(sheet_items(order[:k]), out / "top_outliers.png")
    _contact_sheet(sheet_items(order[-min(6, len(order)) :]), out / "most_typical.png")

    metrics = {
        "config": asdict(cfg),
        "n_scans_scored": len(fm.scan_ids),
        "n_scans_dropped": len(fm.dropped),
        "dropped": fm.dropped,
        "n_features": int(fm.X.shape[1]),
        "image_slots": [b[0] for b in fm.blocks],
        "explained_variance_ratio": [round(float(v), 4) for v in s.explained_variance_ratio],
        "score_quantiles": {
            q: round(float(np.quantile(s.score, float(q))), 4)
            for q in ("0.5", "0.9", "0.95", "0.99")
        },
        "score_max": round(float(s.score.max()), 4),
        "batteryct_version": __version__,
        "runtime_s": round(time.time() - t0, 1),
        "note": "Unsupervised, fitted on the whole cohort. Score = differs from cohort, "
        "not defective, until reviewed.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return out / "scores.csv"
