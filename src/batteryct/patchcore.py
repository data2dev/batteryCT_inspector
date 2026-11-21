"""PatchCore for an unlabeled cohort that may contain a few anomalous cells.

Standard PatchCore builds a memory bank of patch features from *known-normal*
images. Here no cell is known to be normal, so three guards keep rare anomalies
from being "explained away" by the bank (see docs/patchcore.md):

1. **Cross-fitting.** Cells are split into ``folds``; each fold is scored against a
   memory bank built only from the other folds, so a cell never matches itself.
2. **k-th nearest neighbour.** A patch's score is its distance to the k-th closest
   bank patch, so a pattern shared by fewer than ~k other cells still scores high.
3. **Trimming.** After a first pass, the top ``trim_top_frac`` of cells are removed
   from all banks and everything is re-scored.

The bank is position-free (patches are compared regardless of where they sit), so
a cell rotated in the scanner is not penalized for its tab being elsewhere.

Planted anomalies (bright particles, dark voids, thin gaps) are added to *copies*
of real cells. Copies are only ever queries, never bank members, and are scored
against the same bank as their source cell's fold (which excludes the source), so
they measure recall without touching the real ranking.

Heavy lifting (feature extraction, k-NN) uses PyTorch when available (GPU on
Kaggle); the k-NN also has a NumPy fallback so the logic is testable without torch.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from batteryct import __version__

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]

FEATURE_STRIDE = 8  # WideResNet layer2 stride, relative to the (scaled) input


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PatchCoreParams:
    scale: float = 0.25  # image downscale before the CNN (0.25: ~4x fewer pixels per side)
    k: int = 3  # score = distance to the k-th nearest bank patch
    folds: int = 5
    ref_stride: int = 2  # keep every n-th patch row/column in the bank (neighbours overlap)
    trim_top_frac: float = 0.02
    proj_dim: int = 256  # fixed random projection of the 1536-d WRN features
    top_patches: int = 3  # cell score per view type = mean of its top-n patch scores
    hotspots: int = 30  # review pack: top cells, one hotspot each
    inject: int = 40  # planted-anomaly copies (0 disables)
    weights: str = "wide_resnet50_2-95faca4d.pth"
    limit_cells: int | None = None  # smoke runs
    allow_cpu: bool = False  # run even if the config asks for a GPU and none is present
    seed: int = 0

    @classmethod
    def from_extra(cls, extra: dict[str, Any]) -> PatchCoreParams:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(extra) - known
        if unknown:
            raise ValueError(f"unknown patchcore params: {sorted(unknown)}")
        p = cls(**extra)
        if not 0.05 <= p.scale <= 1.0:
            raise ValueError("scale must be in [0.05, 1]")
        if p.k < 1 or p.folds < 2 or p.ref_stride < 1 or p.proj_dim < 8:
            raise ValueError("k >= 1, folds >= 2, ref_stride >= 1, proj_dim >= 8 required")
        if not 0.0 <= p.trim_top_frac < 0.5:
            raise ValueError("trim_top_frac must be in [0, 0.5)")
        return p


# ------------------------------------------------------------------- patch storage


@dataclass
class PatchTable:
    """All patch features of one view type (axial or radial), across all cells.

    ``feats[i]`` belongs to cell ``cell[i]``, image slot ``slot[i]``, grid ``(y[i], x[i])``.
    """

    feats: FloatArray  # (N, D), float16 storage is fine
    cell: IntArray
    slot: IntArray
    y: IntArray
    x: IntArray


def stack_tables(parts: Sequence[PatchTable]) -> PatchTable:
    return PatchTable(
        np.concatenate([p.feats for p in parts]),
        np.concatenate([p.cell for p in parts]),
        np.concatenate([p.slot for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate([p.x for p in parts]),
    )


def grid_table(feat: FloatArray, cell: int, slot: int) -> PatchTable:
    """Flatten an (H, W, D) feature grid into a PatchTable."""
    h, w, d = feat.shape
    yy, xx = np.mgrid[0:h, 0:w]
    n = h * w
    return PatchTable(
        feat.reshape(n, d),
        np.full(n, cell, dtype=np.int64),
        np.full(n, slot, dtype=np.int64),
        yy.ravel().astype(np.int64),
        xx.ravel().astype(np.int64),
    )


# ----------------------------------------------------------------------------- k-NN

KnnFn = Callable[[FloatArray, FloatArray, int], tuple[FloatArray, IntArray]]


def knn_numpy(
    query: FloatArray, ref: FloatArray, k: int, chunk: int = 2048
) -> tuple[FloatArray, IntArray]:
    """Distances to and indices of the k nearest rows of ``ref`` (Euclidean, float64 math)."""
    k = min(k, len(ref))
    r = ref.astype(np.float64)
    rn = (r**2).sum(1)
    dists = np.empty((len(query), k), np.float32)
    idx = np.empty((len(query), k), np.int64)
    for s in range(0, len(query), chunk):
        q = query[s : s + chunk].astype(np.float64)
        d2 = (q**2).sum(1)[:, None] + rn[None, :] - 2.0 * q @ r.T
        part = np.argpartition(d2, k - 1, axis=1)[:, :k]
        pd = np.take_along_axis(d2, part, 1)
        order = np.argsort(pd, axis=1)
        idx[s : s + chunk] = np.take_along_axis(part, order, 1)
        dists[s : s + chunk] = np.sqrt(np.maximum(np.take_along_axis(pd, order, 1), 0.0))
    return dists, idx


def make_knn_torch(device: str, q_chunk: int = 2048, r_chunk: int = 262_144) -> KnnFn:
    """k-NN on GPU (or CPU) with float32 math, chunked over queries and bank."""
    import torch

    def knn(query: FloatArray, ref: FloatArray, k: int) -> tuple[FloatArray, IntArray]:
        k = min(k, len(ref))
        R = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32)).to(device)
        rn = (R * R).sum(1)
        out_d = np.empty((len(query), k), np.float32)
        out_i = np.empty((len(query), k), np.int64)
        with torch.no_grad():
            for s in range(0, len(query), q_chunk):
                Q = torch.from_numpy(np.ascontiguousarray(query[s : s + q_chunk], np.float32))
                Q = Q.to(device)
                qn = (Q * Q).sum(1, keepdim=True)
                best_d = torch.full((len(Q), k), float("inf"), device=device)
                best_i = torch.zeros((len(Q), k), dtype=torch.long, device=device)
                for r0 in range(0, len(R), r_chunk):
                    Rc = R[r0 : r0 + r_chunk]
                    d2 = qn + rn[r0 : r0 + r_chunk][None, :] - 2.0 * (Q @ Rc.T)
                    kk = min(k, Rc.shape[0])
                    d, i = torch.topk(d2, kk, dim=1, largest=False)
                    cat_d = torch.cat([best_d, d], 1)
                    cat_i = torch.cat([best_i, i + r0], 1)
                    best_d, sel = torch.topk(cat_d, k, dim=1, largest=False)
                    best_i = torch.gather(cat_i, 1, sel)
                out_d[s : s + len(Q)] = best_d.clamp_min(0).sqrt().cpu().numpy()
                out_i[s : s + len(Q)] = best_i.cpu().numpy()
        return out_d, out_i

    return knn


# ------------------------------------------------------------------------- scoring


def assign_folds(n_cells: int, folds: int, seed: int) -> IntArray:
    """Balanced, seeded random fold per cell."""
    rng = np.random.default_rng(seed)
    f = np.arange(n_cells) % folds
    rng.shuffle(f)
    return f.astype(np.int64)


@dataclass
class KindScores:
    patch: FloatArray  # per query patch: k-th NN distance
    nn: IntArray  # per query patch: index (into the full table) of its nearest bank patch


def score_kind(
    table: PatchTable,
    cell_fold: IntArray,
    is_real: npt.NDArray[np.bool_],
    excluded: set[int],
    p: PatchCoreParams,
    knn: KnnFn,
) -> KindScores:
    """Cross-fitted k-NN scores for every patch of one view type.

    Bank for fold f: real cells not in fold f and not ``excluded``, spatially
    subsampled by ``ref_stride``. Queries: every patch of cells in fold f (real
    cells and planted copies alike; copies carry their source cell's fold).
    """
    fold_of_patch = cell_fold[table.cell]
    real_patch = is_real[table.cell]
    keep = (table.y % p.ref_stride == 0) & (table.x % p.ref_stride == 0)
    if excluded:
        keep &= ~np.isin(table.cell, np.fromiter(excluded, dtype=np.int64))
    patch = np.zeros(len(table.feats), np.float32)
    nn = np.zeros(len(table.feats), np.int64)
    for f in range(int(cell_fold.max()) + 1):
        bank = np.flatnonzero(real_patch & keep & (fold_of_patch != f))
        query = np.flatnonzero(fold_of_patch == f)
        if len(query) == 0:
            continue
        if len(bank) == 0:
            raise ValueError("empty memory bank; too few cells for the number of folds")
        d, i = knn(table.feats[query], table.feats[bank], p.k)
        patch[query] = d[:, -1]
        nn[query] = bank[i[:, 0]]
    return KindScores(patch, nn)


def cell_scores(
    kinds: dict[str, PatchTable],
    scores: dict[str, KindScores],
    n_cells: int,
    is_real: npt.NDArray[np.bool_],
    top: int,
) -> tuple[FloatArray, dict[str, FloatArray], dict[str, IntArray]]:
    """Per cell: mean of the top-n patch scores per view type, divided by that view
    type's median over real cells; overall score = max over view types.

    Also returns, per view type, the table index of each cell's highest patch.
    """
    per_kind: dict[str, FloatArray] = {}
    argmax: dict[str, IntArray] = {}
    for kind, t in kinds.items():
        s = scores[kind].patch
        raw = np.zeros(n_cells, np.float32)
        best = np.full(n_cells, -1, np.int64)
        order = np.argsort(t.cell, kind="stable")
        bounds = np.searchsorted(t.cell[order], np.arange(n_cells + 1))
        for c in range(n_cells):
            ids = order[bounds[c] : bounds[c + 1]]
            if len(ids) == 0:
                continue
            vals = s[ids]
            n = min(top, len(vals))
            topn = np.argpartition(-vals, n - 1)[:n]
            raw[c] = vals[topn].mean()
            best[c] = ids[int(np.argmax(vals))]
        med = float(np.median(raw[is_real])) or 1.0
        per_kind[kind] = raw / med
        argmax[kind] = best
    overall = np.max(np.stack(list(per_kind.values())), axis=0)
    return overall, per_kind, argmax


# ------------------------------------------------------------------- planting


ANOMALY_TYPES = ("particle", "void", "gap")


@dataclass(frozen=True)
class Plant:
    """One planted anomaly, in prepared-image pixel coordinates (full resolution)."""

    source_cell: int
    slot: int
    kind: str
    cy: int
    cx: int
    size_px: int  # radius (particle, void) or half-length (gap)


def plan_plants(
    n: int,
    n_cells: int,
    n_slots: int,
    masks: Callable[[int, int], npt.NDArray[np.bool_]],
    seed: int,
) -> list[Plant]:
    """Choose source cells, slots, types, sizes and positions inside the cell body."""
    rng = np.random.default_rng(seed + 1)
    plants: list[Plant] = []
    sources = rng.choice(n_cells, size=min(n, n_cells), replace=n > n_cells)
    for j, c in enumerate(sources):
        slot = int(rng.integers(n_slots))
        kind = ANOMALY_TYPES[j % len(ANOMALY_TYPES)]
        mask = masks(int(c), slot)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            continue
        pick = int(rng.integers(len(ys)))
        size = int(rng.choice([6, 10, 16, 24, 36]))  # ~0.1-0.5 mm at ~14 um/px
        plants.append(Plant(int(c), slot, kind, int(ys[pick]), int(xs[pick]), size))
    return plants


def apply_plant(img: npt.NDArray[np.float32], pl: Plant) -> npt.NDArray[np.float32]:
    """Return a copy of a prepared [0, 1] image with the anomaly drawn in."""
    out = img.copy()
    h, w = out.shape
    yy, xx = np.ogrid[0:h, 0:w]
    hi = max(float(np.percentile(img, 99.9)), 0.8)  # at least as bright as the can wall
    lo = min(float(np.percentile(img, 1.0)), 0.05)  # background-dark, never mid-grey
    if pl.kind == "particle":  # dense debris: brighter than anything else in the roll
        m = (yy - pl.cy) ** 2 + (xx - pl.cx) ** 2 <= pl.size_px**2
        out[m] = min(1.0, hi * 1.1)
    elif pl.kind == "void":  # gas pocket: background-dark ellipse
        m = ((yy - pl.cy) / pl.size_px) ** 2 + ((xx - pl.cx) / (0.6 * pl.size_px)) ** 2 <= 1
        out[m] = lo
    else:  # gap: thin dark line across the layers
        t = max(1, pl.size_px // 8)
        m = (np.abs(yy - pl.cy) <= t) & (np.abs(xx - pl.cx) <= pl.size_px)
        out[m] = lo
    return out


# ---------------------------------------------------------------- run orchestration


@dataclass
class CellSlots:
    scan_id: str
    paths: list[Path]


def run_patchcore(
    cfg: Any,  # ExperimentConfig (kept loose to avoid a circular import)
    cells: list[CellSlots],
    slot_labels: list[str],
    meta: dict[str, dict[str, str]],
    extract: Callable[[npt.NDArray[np.float32], str], FloatArray],
    prepare: Callable[[Path, str], npt.NDArray[np.float32]],
    knn: KnnFn,
    out: Path,
    p: PatchCoreParams,
    log: Callable[[str], None] = print,
) -> Path:
    """Score cells with cross-fitted PatchCore and write outputs into ``out``.

    ``prepare(path, view)`` returns the full-resolution prepared image in [0, 1];
    ``extract(image, view)`` returns an (H, W, D) feature grid for it.
    """
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    kinds_of_slot = [lab.split("[")[0] for lab in slot_labels]
    n_real = len(cells)

    parts: dict[str, list[PatchTable]] = {k: [] for k in sorted(set(kinds_of_slot))}
    grid_hw: dict[tuple[int, int], tuple[int, int]] = {}
    for c, cell in enumerate(cells):
        for s, path in enumerate(cell.paths):
            g = extract(prepare(path, kinds_of_slot[s]), kinds_of_slot[s])
            grid_hw[(c, s)] = g.shape[:2]
            parts[kinds_of_slot[s]].append(grid_table(g.astype(np.float16), c, s))
        if (c + 1) % 25 == 0 or c + 1 == n_real:
            log(f"[patchcore] features {c + 1}/{n_real} cells ({time.time() - t0:.0f}s)")

    # Planted copies: cell ids n_real, n_real+1, ... ; only the planted slot differs.
    plants: list[Plant] = []
    if p.inject:

        def body_mask(c: int, s: int) -> npt.NDArray[np.bool_]:
            img = prepare(cells[c].paths[s], kinds_of_slot[s])
            m = img > 0.25
            h, w = m.shape
            m[: h // 8] = m[-h // 8 :] = False  # keep away from caps / outer edge
            m[:, : w // 8] = m[:, -w // 8 :] = False
            return m

        plants = plan_plants(p.inject, n_real, len(slot_labels), body_mask, p.seed)
        for j, pl in enumerate(plants):
            cid = n_real + j
            for s in range(len(slot_labels)):
                kind = kinds_of_slot[s]
                if s == pl.slot:
                    img = apply_plant(prepare(cells[pl.source_cell].paths[s], kind), pl)
                    g = extract(img, kind)
                    parts[kind].append(grid_table(g.astype(np.float16), cid, s))
                else:  # identical to the source cell: reuse its patches
                    src = next(
                        t for t in parts[kind] if t.cell[0] == pl.source_cell and t.slot[0] == s
                    )
                    parts[kind].append(
                        PatchTable(src.feats, np.full_like(src.cell, cid), src.slot, src.y, src.x)
                    )
                grid_hw[(cid, s)] = grid_hw[(pl.source_cell, s)]
        log(f"[patchcore] planted {len(plants)} anomalies into copies")

    tables = {k: stack_tables(v) for k, v in parts.items()}
    n_all = n_real + len(plants)
    is_real = np.zeros(n_all, bool)
    is_real[:n_real] = True
    fold = np.empty(n_all, np.int64)
    fold[:n_real] = assign_folds(n_real, p.folds, p.seed)
    for j, pl in enumerate(plants):
        fold[n_real + j] = fold[pl.source_cell]

    excluded: set[int] = set()
    passes = 2 if p.trim_top_frac > 0 else 1
    for pas in range(passes):
        scores = {k: score_kind(t, fold, is_real, excluded, p, knn) for k, t in tables.items()}
        overall, per_kind, best = cell_scores(tables, scores, n_all, is_real, p.top_patches)
        log(f"[patchcore] pass {pas + 1}/{passes} done ({time.time() - t0:.0f}s)")
        if pas + 1 < passes:
            n_trim = round(p.trim_top_frac * n_real)
            excluded = set(np.argsort(-overall[:n_real])[:n_trim].tolist())

    # ---------------- outputs: scores.csv
    kinds = sorted(tables)
    order = np.argsort(-overall[:n_real])

    def hotspot(c: int) -> tuple[str, int, int, int, int]:
        """(view kind, table row, slot, y, x) of a cell's strongest patch overall."""
        kind = max(kinds, key=lambda k: per_kind[k][c])
        row = int(best[kind][c])
        t = tables[kind]
        return kind, row, int(t.slot[row]), int(t.y[row]), int(t.x[row])

    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "scan_id",
                "score",
                *[f"{k}_score" for k in kinds],
                "hotspot_view",
                "hotspot_y_px",
                "hotspot_x_px",
                "in_inventory",
                "cell_serial_number",
            ]
        )
        for rank, c in enumerate(order, 1):
            kind, _, slot, gy, gx = hotspot(int(c))
            py, px = _grid_to_px(gy, gx, p.scale)
            sid = cells[c].scan_id
            w.writerow(
                [
                    rank,
                    sid,
                    f"{overall[c]:.4f}",
                    *[f"{per_kind[k][c]:.4f}" for k in kinds],
                    slot_labels[slot],
                    py,
                    px,
                    meta[sid].get("in_inventory", ""),
                    meta[sid].get("cell_serial_number", ""),
                ]
            )

    # ---------------- planted-anomaly recall
    plant_rows: list[dict[str, Any]] = []
    real_scores = overall[:n_real]
    thr = {q: float(np.quantile(real_scores, q)) for q in (0.9, 0.95, 0.99)}
    patch_thr = {
        k: float(np.quantile(scores[k].patch[is_real[tables[k].cell]], 0.999)) for k in kinds
    }
    for j, pl in enumerate(plants):
        cid = n_real + j
        kind = kinds_of_slot[pl.slot]
        t = tables[kind]
        rows = np.flatnonzero((t.cell == cid) & (t.slot == pl.slot))
        py, px = _grid_to_px(t.y[rows], t.x[rows], p.scale)
        reach = pl.size_px + FEATURE_STRIDE / p.scale  # within the anomaly or one patch of it
        near = (np.abs(py - pl.cy) <= reach) & (np.abs(px - pl.cx) <= reach)
        local_max = float(scores[kind].patch[rows[near]].max()) if near.any() else 0.0
        src_cell = pl.source_cell
        plant_rows.append(
            {
                "copy_of": cells[src_cell].scan_id,
                "slot": slot_labels[pl.slot],
                "type": pl.kind,
                "size_px": pl.size_px,
                "y_px": pl.cy,
                "x_px": pl.cx,
                "copy_score": round(float(overall[cid]), 4),
                "source_score": round(float(overall[src_cell]), 4),
                "copy_percentile": round(float((real_scores < overall[cid]).mean() * 100), 1),
                "cell_detected_p95": bool(overall[cid] > thr[0.95]),
                "cell_detected_p99": bool(overall[cid] > thr[0.99]),
                "located": bool(local_max > patch_thr[kind]),
            }
        )
    if plant_rows:
        with (out / "planted.csv").open("w", newline="") as f:
            dw = csv.DictWriter(f, fieldnames=list(plant_rows[0]))
            dw.writeheader()
            dw.writerows(plant_rows)

    def recall(key: str, rows: list[dict[str, Any]]) -> float | None:
        return round(float(np.mean([r[key] for r in rows])), 3) if rows else None

    by_type = {
        typ: {
            k: recall(k, [r for r in plant_rows if r["type"] == typ])
            for k in ("cell_detected_p95", "cell_detected_p99", "located")
        }
        for typ in ANOMALY_TYPES
    }
    by_size = {
        str(sz): {
            k: recall(k, [r for r in plant_rows if r["size_px"] == sz])
            for k in ("cell_detected_p95", "located")
        }
        for sz in sorted({r["size_px"] for r in plant_rows})
    }

    # ---------------- review pack: hotspot vs nearest match in another cell
    review = out / "review"
    review.mkdir(exist_ok=True)
    lines = [
        "# PatchCore review pack",
        "",
        "Each page: where the cell's most unusual patch sits (left, box), a full-resolution",
        "crop of it (middle), and the **most similar patch found in any other cell** (right).",
        "Ask only: *is the middle crop genuinely unlike the right one?* A high score means",
        "*unlike the rest of the cohort*, not *defective*.",
        "",
        "Images derived from the Glimpse battery CT dataset (CC BY-NC-SA 4.0), normalized "
        "and cropped.",
        "",
        "| rank | scan | serial | score | "
        + " | ".join(f"{k} score" for k in kinds)
        + " | hotspot | matched with | page |",
        "|---|---|---|---|" + "---|" * len(kinds) + "---|---|---|",
    ]
    for rank, c in enumerate(order[: p.hotspots], 1):
        c = int(c)
        kind, row, slot, gy, gx = hotspot(c)
        t = tables[kind]
        mrow = int(scores[kind].nn[row])
        mc, ms = int(t.cell[mrow]), int(t.slot[mrow])
        my, mx = _grid_to_px(int(t.y[mrow]), int(t.x[mrow]), p.scale)
        py, px = _grid_to_px(gy, gx, p.scale)
        img = prepare(cells[c].paths[slot], kind)
        match = prepare(cells[mc].paths[ms], kind)
        name = f"{rank:02d}_{cells[c].scan_id}.jpg"
        _hotspot_page(
            img,
            (py, px),
            match,
            (my, mx),
            p.scale,
            f"#{rank} scan {cells[c].scan_id}  {slot_labels[slot]}  score {overall[c]:.2f}",
            f"closest match: scan {cells[mc].scan_id} {slot_labels[ms]}",
            review / name,
        )
        sid = cells[c].scan_id
        serial = meta[sid].get("cell_serial_number") or "(not in inventory)"
        lines.append(
            f"| {rank} | {sid} | {serial} | {overall[c]:.2f} | "
            + " | ".join(f"{per_kind[k][c]:.2f}" for k in kinds)
            + f" | {slot_labels[slot]} ({py},{px}) | {cells[mc].scan_id} {slot_labels[ms]}"
            f" | [{name}]({name}) |"
        )
    (review / "index.md").write_text("\n".join(lines) + "\n")

    metrics = {
        "method": "patchcore",
        "params": asdict(p),
        "n_cells_scored": n_real,
        "n_patches": {k: len(t.feats) for k, t in tables.items()},
        "trimmed_cells": sorted(cells[c].scan_id for c in excluded),
        "score_quantiles": {str(q): round(v, 4) for q, v in thr.items()},
        "score_max": round(float(real_scores.max()), 4),
        "planted": {
            "n": len(plant_rows),
            "cell_detected_p95": recall("cell_detected_p95", plant_rows),
            "cell_detected_p99": recall("cell_detected_p99", plant_rows),
            "located": recall("located", plant_rows),
            "by_type": by_type,
            "by_size_px": by_size,
            "note": "Recall on synthetic anomalies planted into copies of real cells; "
            "copies are scored against banks that exclude their source cell.",
        },
        "batteryct_version": __version__,
        "runtime_s": round(time.time() - t0, 1),
        "note": "Unsupervised and cross-fitted. Score = unlike the rest of the cohort, "
        "not defective, until reviewed.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return out / "scores.csv"


def _grid_to_px(gy: Any, gx: Any, scale: float) -> tuple[Any, Any]:
    """Feature-grid cell -> centre pixel in the full-resolution prepared image."""
    f = FEATURE_STRIDE / scale
    cy = np.round((np.asarray(gy) + 0.5) * f).astype(np.int64)
    cx = np.round((np.asarray(gx) + 0.5) * f).astype(np.int64)
    if cy.ndim == 0:
        return int(cy), int(cx)
    return cy, cx


def _crop(img: npt.NDArray[np.float32], cy: int, cx: int, half: int) -> Image.Image:
    pad = np.pad(img, half, mode="constant")
    c = pad[cy : cy + 2 * half, cx : cx + 2 * half]
    return Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8))


def _hotspot_page(
    img: npt.NDArray[np.float32],
    at: tuple[int, int],
    match: npt.NDArray[np.float32],
    match_at: tuple[int, int],
    scale: float,
    title: str,
    match_title: str,
    out: Path,
) -> None:
    """[context with box | full-res crop | nearest-match crop], each crop 3 patches wide."""
    half = math.ceil(1.5 * FEATURE_STRIDE / scale)  # crop spans ~3 feature patches
    disp = 360
    ctx_h = 480
    ctx = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    sx = ctx_h / ctx.height
    ctx = ctx.resize((max(1, int(ctx.width * sx)), ctx_h), Image.Resampling.BILINEAR).convert("RGB")
    d = ImageDraw.Draw(ctx)
    y0, x0 = at[0] * sx, at[1] * sx
    r = max(4.0, half * sx)
    d.rectangle([x0 - r, y0 - r, x0 + r, y0 + r], outline=(255, 40, 40), width=2)
    crop = _crop(img, *at, half).resize((disp, disp), Image.Resampling.NEAREST).convert("RGB")
    mcrop = _crop(match, *match_at, half).resize((disp, disp), Image.Resampling.NEAREST)
    W = ctx.width + 2 * disp + 40
    page = Image.new("RGB", (W, ctx_h + 40), (255, 255, 255))
    page.paste(ctx, (0, 36))
    page.paste(crop, (ctx.width + 20, 36))
    page.paste(mcrop.convert("RGB"), (ctx.width + disp + 40, 36))
    dd = ImageDraw.Draw(page)
    dd.text((4, 4), title, fill=(0, 0, 0))
    dd.text(
        (ctx.width + 20, 20), f"hotspot ({at[0]},{at[1]}) px, {2 * half} px wide", fill=(60, 60, 60)
    )
    dd.text((ctx.width + disp + 40, 20), match_title, fill=(60, 60, 60))
    page.save(out, quality=88)
