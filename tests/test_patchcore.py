"""PatchCore logic on SYNTHETIC data (NumPy path; no torch or real CT data needed)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from batteryct.patchcore import (
    CellSlots,
    PatchCoreParams,
    PatchTable,
    Plant,
    apply_plant,
    assign_folds,
    grid_table,
    knn_numpy,
    run_patchcore,
    score_kind,
    stack_tables,
)
from batteryct.synthetic import DefectKind, make_cell


def test_knn_numpy_matches_brute_force():
    rng = np.random.default_rng(0)
    q, r = rng.normal(size=(50, 8)), rng.normal(size=(300, 8))
    d, i = knn_numpy(q.astype(np.float32), r.astype(np.float32), 4, chunk=7)
    full = np.sqrt(((q[:, None] - r[None]) ** 2).sum(-1))
    want = np.sort(full, 1)[:, :4]
    np.testing.assert_allclose(d, want, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(full[np.arange(50)[:, None], i], want, rtol=1e-5, atol=1e-5)


def test_knn_torch_matches_numpy():
    pytest.importorskip("torch")
    from batteryct.patchcore import make_knn_torch

    rng = np.random.default_rng(1)
    q = rng.normal(size=(300, 16)).astype(np.float32)
    r = rng.normal(size=(1000, 16)).astype(np.float32)
    d1, i1 = knn_numpy(q, r, 3)
    d2, i2 = make_knn_torch("cpu", q_chunk=64, r_chunk=128)(q, r, 3)
    np.testing.assert_allclose(d1, d2, rtol=1e-4, atol=1e-4)
    assert (i1 == i2).mean() > 0.99


def test_assign_folds_balanced_and_seeded():
    f = assign_folds(103, 5, seed=3)
    counts = np.bincount(f)
    assert counts.max() - counts.min() <= 1
    np.testing.assert_array_equal(f, assign_folds(103, 5, seed=3))


def _table(n_cells: int, rng: np.random.Generator, per_cell: int = 30) -> PatchTable:
    parts = [
        grid_table(rng.normal(size=(1, per_cell, 4)).astype(np.float32), c, 0)
        for c in range(n_cells)
    ]
    return stack_tables(parts)


def test_cross_fitting_keeps_unique_patch_high():
    """A one-off patch must not be matched by itself."""
    rng = np.random.default_rng(0)
    t = _table(20, rng)
    t.feats[5] = 50.0  # cell 0, one wild patch
    folds = assign_folds(20, 4, 0)
    p = PatchCoreParams(k=1, folds=4, ref_stride=1)
    s = score_kind(t, folds, np.ones(20, bool), set(), p, knn_numpy)
    assert s.patch[5] > 10 * np.median(s.patch)


def test_kth_neighbour_catches_pattern_shared_by_two_cells():
    """With k=1 a pattern in two cells (different folds) matches; with k=3 it doesn't."""
    rng = np.random.default_rng(0)
    t = _table(20, rng)
    folds = assign_folds(20, 4, 0)
    a, b = 0, int(np.flatnonzero(folds != folds[0])[0])
    rows_a = np.flatnonzero(t.cell == a)[:1]
    rows_b = np.flatnonzero(t.cell == b)[:1]
    t.feats[rows_a] = t.feats[rows_b] = 50.0
    real = np.ones(20, bool)
    s1 = score_kind(t, folds, real, set(), PatchCoreParams(k=1, folds=4, ref_stride=1), knn_numpy)
    s3 = score_kind(t, folds, real, set(), PatchCoreParams(k=3, folds=4, ref_stride=1), knn_numpy)
    assert s1.patch[rows_a[0]] < 1.0  # found its twin in the other cell
    assert s3.patch[rows_a[0]] > 10 * np.median(s3.patch)


def test_params_validation():
    assert PatchCoreParams.from_extra({"k": 5}).k == 5
    with pytest.raises(ValueError):
        PatchCoreParams.from_extra({"nope": 1})
    with pytest.raises(ValueError):
        PatchCoreParams.from_extra({"folds": 1})


@pytest.mark.parametrize("kind", ["particle", "void", "gap"])
def test_apply_plant_changes_only_local_region(kind):
    img = np.full((100, 100), 0.4, np.float32)
    img[0, 0] = 1.0
    out = apply_plant(img, Plant(0, 0, kind, 50, 50, 10))
    changed = np.argwhere(out != img)
    assert len(changed) > 0
    assert np.abs(changed - 50).max() <= 10
    assert img[50, 50] == pytest.approx(0.4)  # input untouched


# --- end to end with a stand-in feature extractor --------------------------------


def _fake_extract(img: np.ndarray, view: str) -> np.ndarray:
    """4x4-pixel blocks of an 8x-downsampled image -> 16-d features (stride 8)."""
    h, w = (img.shape[0] // 8) * 8, (img.shape[1] // 8) * 8
    small = img[:h, :w].reshape(h // 8, 8, w // 8, 8).mean((1, 3))
    pad = np.pad(small, 2, mode="edge")
    gh, gw = small.shape
    feats = np.stack(
        [pad[dy : dy + gh, dx : dx + gw] for dy in range(4) for dx in range(4)], axis=-1
    )
    return feats.astype(np.float32)


def test_run_patchcore_end_to_end(tmp_path: Path) -> None:
    n = 24
    images: dict[Path, np.ndarray] = {}
    cells = []
    for c in range(n):
        paths = []
        for s in range(2):
            p = tmp_path / f"{c}_{s}.png"
            defect: DefectKind = "void" if (c == 7 and s == 1) else "none"
            images[p] = make_cell(128, defect=defect, seed=c * 3 + s, noise=0.01).image
            paths.append(p)
        cells.append(CellSlots(str(1000 + c), paths))
    meta = {cs.scan_id: {"in_inventory": "true", "cell_serial_number": "SN"} for cs in cells}
    params = PatchCoreParams(scale=1.0, k=2, folds=4, ref_stride=1, inject=6, hotspots=3)
    out = tmp_path / "out"
    run_patchcore(
        None,
        cells,
        ["radial[0]", "radial[1]"],
        meta,
        extract=_fake_extract,
        prepare=lambda path, view: images[path],
        knn=knn_numpy,
        out=out,
        p=params,
        log=lambda msg: None,
    )
    rows = list(csv.DictReader((out / "scores.csv").open()))
    assert len(rows) == n
    assert rows[0]["scan_id"] == "1007"  # the cell with the real (synthetic) void
    planted = list(csv.DictReader((out / "planted.csv").open()))
    assert len(planted) == 6
    m = json.loads((out / "metrics.json").read_text())
    assert m["planted"]["n"] == 6 and m["n_cells_scored"] == n
    assert len(m["trimmed_cells"]) == round(0.02 * n)
    pages = sorted(p.name for p in (out / "review").glob("*.jpg"))
    assert len(pages) == 3
    assert "most similar patch" in (out / "review" / "index.md").read_text()


def test_gpu_config_refuses_cpu(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    if torch.cuda.is_available():
        pytest.skip("a GPU is present")
    from batteryct.experiment import ExperimentConfig, run_experiment

    (tmp_path / "manifest.csv").write_text("scan_id,view,position,file\n")
    cfg = ExperimentConfig.from_mapping(
        {"name": "g", "cohort": "c", "method": "patchcore", "gpu": True}
    )
    with pytest.raises(RuntimeError, match="phone-verified"):
        run_experiment(cfg, tmp_path, tmp_path / "out")
