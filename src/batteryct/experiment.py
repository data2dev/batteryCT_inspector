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
import os
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
METHODS = ("pca_residual", "patchcore")


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
    balance_views: bool = True
    exclude_not_in_inventory: bool = False
    review_top: int = 15
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
            balance_views=bool(p.pop("balance_views", True)),
            exclude_not_in_inventory=bool(p.pop("exclude_not_in_inventory", False)),
            review_top=int(p.pop("review_top", 15)),
            extra=p,
        )
        if method == "patchcore":
            from batteryct.patchcore import PatchCoreParams

            PatchCoreParams.from_extra(cfg.extra)  # fail fast on bad params
        if cfg.size < 8 or cfg.n_components < 1 or cfg.top_k < 1 or cfg.review_top < 0:
            raise ValueError("size >= 8, n_components >= 1, top_k >= 1, review_top >= 0 required")
        return cfg


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class ImageRef:
    scan_id: str
    view: str
    position: float
    path: Path


def _walk_files(root: Path, name: str | None = None, suffix: str | None = None) -> list[Path]:
    """Recursive file search that follows symlinked folders (Kaggle mounts use them)."""
    found = []
    for dirpath, _, files in os.walk(root, followlinks=True):
        for f in files:
            if (name is None or f == name) and (suffix is None or f.endswith(suffix)):
                found.append(Path(dirpath) / f)
    return found


def find_manifest(data_dir: str | Path) -> Path:
    """Locate ``manifest.csv`` at or below ``data_dir`` (Kaggle may nest folders)."""
    root = Path(data_dir)
    if (root / "manifest.csv").is_file():
        return root / "manifest.csv"
    hits = sorted(_walk_files(root, name="manifest.csv"), key=lambda p: (len(p.parts), str(p)))
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
                index = {
                    "/".join(q.parts[-2:]): q for q in _walk_files(Path(data_dir), suffix=".png")
                }
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


def grid_shape(view: str, size: int) -> tuple[int, int]:
    """(width, height) of a view's feature grid. Axial slices are long and thin."""
    return (size, size) if view == "radial" else (size, 4 * size)


def prepared_image(path: Path, view: str) -> Image.Image:
    """Normalized 8-bit image as the model sees it (radial views cropped around the cell)."""
    img = normalize_percentile(load_image(path))
    if view == "radial":
        side = min(img.shape)
        img = center_crop(img, side, center=foreground_centroid(img, threshold=0.15))
    return Image.fromarray((img * 255).astype(np.uint8))


def image_features(path: Path, view: str, size: int) -> FloatArray:
    """Downsample one view to a fixed grid (see :func:`grid_shape`)."""
    small = prepared_image(path, view).resize(grid_shape(view, size), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.float64).ravel() / 255.0


@dataclass
class FeatureMatrix:
    scan_ids: list[str]
    X: FloatArray
    blocks: list[tuple[str, int, int]]  # (label, start, stop) per image slot
    slot_paths: list[list[Path]]  # per scan, the image path of each slot
    dropped: dict[str, str]


@dataclass
class CohortSlots:
    scan_ids: list[str]
    slot_labels: list[str]  # e.g. ["axial[0]", ..., "radial[2]"]
    paths: list[list[Path]]  # per scan, one path per slot
    dropped: dict[str, str]


def cohort_slots(
    refs: list[ImageRef], cfg: ExperimentConfig, exclude: set[str] | None = None
) -> CohortSlots:
    """Fixed image-slot layout (most common views x counts); scans that differ are dropped."""
    by_scan: dict[str, dict[str, list[ImageRef]]] = defaultdict(lambda: defaultdict(list))
    for r in refs:
        if r.view in cfg.views and r.scan_id not in (exclude or set()):
            by_scan[r.scan_id][r.view].append(r)
    layouts: dict[tuple[tuple[str, int], ...], int] = defaultdict(int)
    for views in by_scan.values():
        layouts[tuple((v, len(views.get(v, []))) for v in cfg.views)] += 1
    if not layouts:
        raise ValueError(f"no images for views {cfg.views}")
    layout = max(layouts, key=lambda k: layouts[k])
    labels = [f"{v}[{i}]" for v, n in layout for i in range(n)]
    out = CohortSlots([], labels, [], {})
    for sid in sorted(by_scan):
        views = by_scan[sid]
        if tuple((v, len(views.get(v, []))) for v in cfg.views) != layout:
            out.dropped[sid] = "image count differs from cohort layout"
            continue
        out.scan_ids.append(sid)
        out.paths.append(
            [r.path for v, _ in layout for r in sorted(views[v], key=lambda r: r.position)]
        )
    if len(out.scan_ids) < 3:
        raise ValueError(f"need at least 3 complete scans, got {len(out.scan_ids)}")
    return out


def build_features(
    refs: list[ImageRef], cfg: ExperimentConfig, exclude: set[str] | None = None
) -> FeatureMatrix:
    by_scan: dict[str, dict[str, list[ImageRef]]] = defaultdict(lambda: defaultdict(list))
    for r in refs:
        if r.view in cfg.views and r.scan_id not in (exclude or set()):
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
    block_residual: FloatArray  # (n_scans, n_blocks), each block divided by its cohort median
    view_score: dict[str, FloatArray]  # per view type, residual / cohort median
    pixel_ratio: FloatArray  # (n_scans, n_features): residual^2 / cohort median, for heatmaps
    explained_variance_ratio: FloatArray


def view_weights(blocks: list[tuple[str, int, int]], n_features: int) -> FloatArray:
    """Per-feature weights giving every view type (axial, radial) equal total influence.

    Without this, the four long axial slices supply ~84% of the features and decide
    nearly every score.
    """
    kinds = sorted({label.split("[")[0] for label, _, _ in blocks})
    cols = {k: sum(b - a for lab, a, b in blocks if lab.split("[")[0] == k) for k in kinds}
    w = np.ones(n_features)
    for lab, a, b in blocks:
        w[a:b] = 1.0 / np.sqrt(len(kinds) * cols[lab.split("[")[0]])
    return np.asarray(w * np.sqrt(n_features))  # keep the overall scale comparable


def robust_standardize(X: FloatArray) -> FloatArray:
    med = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - med), axis=0)
    scale = np.where(mad > 1e-9, mad, X.std(axis=0) + 1e-9)
    return np.asarray((X - med) / scale)


def pca_residual_scores(
    X: FloatArray,
    n_components: int,
    blocks: list[tuple[str, int, int]],
    balance_views: bool = True,
) -> Scores:
    Z = robust_standardize(X)
    Z = np.clip(Z, -20.0, 20.0)  # one wild pixel must not dominate the fit
    if balance_views:
        Z = Z * view_weights(blocks, Z.shape[1])
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
    R2 = R**2
    block_res = np.stack([R2[:, a:b].sum(axis=1) for _, a, b in blocks], axis=1)
    block_res = block_res / np.maximum(np.median(block_res, axis=0), 1e-12)
    view_score: dict[str, FloatArray] = {}
    for kind in sorted({lab.split("[")[0] for lab, _, _ in blocks}):
        idx = [j for j, (lab, _, _) in enumerate(blocks) if lab.split("[")[0] == kind]
        v = np.asarray([R2[:, a:b].sum(axis=1) for _, a, b in (blocks[j] for j in idx)]).sum(0)
        view_score[kind] = v / max(float(np.median(v)), 1e-12)
    pixel_ratio = R2 / np.maximum(np.median(R2, axis=0), 1e-12)
    evr = (S**2) / max(float((S**2).sum()), 1e-12)
    return Scores(score, residual, t2, block_res, view_score, pixel_ratio, evr[:k])


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


REVIEW_AXIAL = (150, 600)  # display (width, height) per axial slice
REVIEW_RADIAL = (300, 300)


def _overlay(base: Image.Image, ratio: FloatArray | None, grid: tuple[int, int]) -> Image.Image:
    """Grayscale view with a red heatmap where the residual is unusually large.

    ``ratio`` is the per-feature residual^2 divided by its cohort median; 1 = typical.
    Red starts at 4x typical and saturates at 64x.
    """
    rgb = base.convert("RGB")
    if ratio is None:
        return rgb
    heat = np.clip((np.log2(np.maximum(ratio, 1e-12)) - 2.0) / 4.0, 0.0, 1.0)
    heat_img = Image.fromarray((heat.reshape(grid[1], grid[0]) * 255).astype(np.uint8))
    alpha = heat_img.resize(base.size, Image.Resampling.BILINEAR)
    red = Image.new("RGB", base.size, (255, 40, 40))
    return Image.composite(red, rgb, alpha.point(lambda v: int(v * 0.6)))


Slot = tuple[str, Path, "FloatArray | None"]


def _review_image(rows: list[tuple[str, list[Slot]]], size: int, out: Path) -> None:
    """One review page: each row is (label, [(slot label, path, ratio or None), ...])."""

    def disp(view: str) -> tuple[int, int]:
        return REVIEW_RADIAL if view == "radial" else REVIEW_AXIAL

    gap, head = 8, 18
    widths = [sum(disp(lab.split("[")[0])[0] + gap for lab, _, _ in r[1]) for r in rows]
    row_h = max(REVIEW_AXIAL[1], REVIEW_RADIAL[1]) + head * 2
    page = Image.new("RGB", (max(widths), row_h * len(rows)), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    for r, (label, slots) in enumerate(rows):
        y0 = r * row_h
        draw.text((4, y0 + 2), label, fill=(0, 0, 0))
        x = 0
        for lab, path, ratio in slots:
            view = lab.split("[")[0]
            base = prepared_image(path, view).resize(disp(view), Image.Resampling.BILINEAR)
            page.paste(_overlay(base, ratio, grid_shape(view, size)), (x, y0 + head * 2))
            draw.text((x + 2, y0 + head), f"{lab} {path.stem}", fill=(60, 60, 60))
            x += disp(view)[0] + gap
    page.save(out, quality=85)


def run_experiment(cfg: ExperimentConfig, data_dir: str | Path, out_dir: str | Path) -> Path:
    """Score all cells; write scores.csv, metrics.json, contact sheets and a review pack."""
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if cfg.method == "patchcore":
        return _run_patchcore(cfg, data_dir, out)
    refs, meta = load_cohort(data_dir)
    excluded = (
        {sid for sid, m in meta.items() if m["in_inventory"] != "true"}
        if cfg.exclude_not_in_inventory
        else set()
    )
    fm = build_features(refs, cfg, exclude=excluded)
    s = pca_residual_scores(fm.X, cfg.n_components, fm.blocks, cfg.balance_views)

    order = np.argsort(-s.score)
    worst_block = s.block_residual.argmax(axis=1)
    kinds = sorted(s.view_score)
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "scan_id",
                "score",
                "residual",
                "t2",
                *[f"{k}_score" for k in kinds],
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
                    *[f"{s.view_score[k][i]:.3f}" for k in kinds],
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

    # Review pack: every view of each top cell with a residual heatmap, above the
    # same views of the most typical cell for comparison.
    n_review = min(cfg.review_top, len(order))
    if n_review:
        review = out / "review"
        review.mkdir(exist_ok=True)
        ref_i = int(order[-1])
        ref_slots: list[Slot] = [
            (lab, fm.slot_paths[ref_i][j], None) for j, (lab, _, _) in enumerate(fm.blocks)
        ]
        lines = [
            "# Review pack",
            "",
            "Top-ranked cells with every view. Red = where the cell differs most from the "
            "cohort (residual >= 4x typical, saturating at 64x). Bottom row of each page: "
            f"the most typical cell ({fm.scan_ids[ref_i]}) for comparison.",
            "",
            "A high score means *differs from the cohort*, not *defective*. Images derived "
            "from the Glimpse battery CT dataset (CC BY-NC-SA 4.0), normalized and cropped.",
            "",
            "| rank | scan | serial | score | "
            + " | ".join(f"{k} score" for k in kinds)
            + " | driving view | page |",
            "|---|---|---|---|" + "---|" * len(kinds) + "---|---|",
        ]
        for rank, i in enumerate(order[:n_review], 1):
            sid = fm.scan_ids[i]
            slots: list[Slot] = [
                (lab, fm.slot_paths[i][j], s.pixel_ratio[i, a:b])
                for j, (lab, a, b) in enumerate(fm.blocks)
            ]
            serial = meta[sid]["cell_serial_number"] or "(not in inventory)"
            name = f"{rank:02d}_{sid}.jpg"
            _review_image(
                [
                    (f"#{rank}  scan {sid}  {serial}  score {s.score[i]:.2f}", slots),
                    (f"reference: most typical cell {fm.scan_ids[ref_i]}", ref_slots),
                ],
                cfg.size,
                review / name,
            )
            lines.append(
                f"| {rank} | {sid} | {serial} | {s.score[i]:.2f} | "
                + " | ".join(f"{s.view_score[k][i]:.2f}" for k in kinds)
                + f" | {fm.blocks[int(worst_block[i])][0]} | [{name}]({name}) |"
            )
        (review / "index.md").write_text("\n".join(lines) + "\n")

    metrics = {
        "config": asdict(cfg),
        "n_scans_scored": len(fm.scan_ids),
        "n_scans_excluded": len(excluded),
        "excluded": sorted(excluded),
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
        "driving_view_counts": {
            kind: int(sum(fm.blocks[int(b)][0].startswith(kind) for b in worst_block))
            for kind in kinds
        },
        "batteryct_version": __version__,
        "runtime_s": round(time.time() - t0, 1),
        "note": "Unsupervised, fitted on the whole cohort. Score = differs from cohort, "
        "not defective, until reviewed.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return out / "scores.csv"


def prepared_array(path: Path, view: str) -> npt.NDArray[np.float32]:
    """Full-resolution prepared image (see :func:`prepared_image`) as float32 in [0, 1]."""
    return np.asarray(prepared_image(path, view), dtype=np.float32) / 255.0


def _run_patchcore(cfg: ExperimentConfig, data_dir: str | Path, out: Path) -> Path:
    from batteryct.features import WrnExtractor, default_device, find_weights
    from batteryct.patchcore import CellSlots, PatchCoreParams, make_knn_torch, run_patchcore

    p = PatchCoreParams.from_extra(cfg.extra)
    refs, meta = load_cohort(data_dir)
    excluded = (
        {sid for sid, m in meta.items() if m["in_inventory"] != "true"}
        if cfg.exclude_not_in_inventory
        else set()
    )
    cs = cohort_slots(refs, cfg, exclude=excluded)
    cells = [CellSlots(sid, paths) for sid, paths in zip(cs.scan_ids, cs.paths, strict=True)]
    if p.limit_cells:
        cells = cells[: p.limit_cells]
    device = default_device()
    weights = find_weights(p.weights, [data_dir, Path.cwd()])
    extractor = WrnExtractor(weights, device, p.scale, p.proj_dim, p.seed)
    print(f"[patchcore] {len(cells)} cells x {len(cs.slot_labels)} views on {device}; {weights}")
    result = run_patchcore(
        cfg,
        cells,
        cs.slot_labels,
        meta,
        extract=extractor,
        prepare=prepared_array,
        knn=make_knn_torch(device),
        out=out,
        p=p,
    )
    m = json.loads((out / "metrics.json").read_text())
    m.update(
        config=asdict(cfg),
        device=device,
        excluded=sorted(excluded),
        dropped=cs.dropped,
        image_slots=cs.slot_labels,
    )
    (out / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    return result
