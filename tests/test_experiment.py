"""Experiment tests on a SYNTHETIC cohort laid out like an ingest output (no real data)."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from batteryct.experiment import (
    ExperimentConfig,
    build_features,
    load_cohort,
    pca_residual_scores,
    run_experiment,
    view_weights,
)
from batteryct.io import save_image
from batteryct.synthetic import make_cell

FIELDS = ["scan_id", "view", "position", "file", "in_inventory", "cell_serial_number"]


def axial_image(seed: int, crack: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((96, 32), 0.4) + rng.normal(0, 0.02, (96, 32))
    img[:, 14:18] = 0.05  # mandrel channel
    if crack:
        img[40:60, 6:10] = 0.95
    return np.clip(img, 0, 1)


def write_cohort(root: Path, n_normal: int = 16, nested: bool = False) -> str:
    """n_normal normal cells plus one with a void (radial) and a bright streak (axial)."""
    base = root / "inner" if nested else root
    rows = []
    for i in range(n_normal + 1):
        sid = f"{1000 + i}"
        odd = i == n_normal
        for pos in (0.0, 90.0):
            rel = f"images/{sid}/axial_{int(pos)}.png"
            p = base / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            save_image(axial_image(i * 7 + int(pos), crack=odd), p)
            rows.append([sid, "axial", pos, rel, "true", f"SN{i}"])
        for k, pos in enumerate((16.0, 32.0)):
            rel = f"images/{sid}/radial_{int(pos)}.png"
            img = make_cell(96, defect="void" if odd else "none", seed=i * 10 + k).image
            save_image(img, base / rel)
            rows.append([sid, "radial", pos, rel, "true", f"SN{i}"])
    with (root / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    return f"{1000 + n_normal}"


CFG: dict[str, Any] = {
    "name": "t",
    "cohort": "na18650",
    "params": {"size": 16, "n_components": 3, "top_k": 4},
}


def test_config_parsing_and_validation():
    cfg = ExperimentConfig.from_mapping(CFG)
    assert cfg.size == 16 and cfg.views == ("axial", "radial")
    with pytest.raises(ValueError):
        ExperimentConfig.from_mapping({"name": "x"})
    with pytest.raises(ValueError):
        ExperimentConfig.from_mapping({**CFG, "method": "magic"})
    with pytest.raises(ValueError):
        ExperimentConfig.from_mapping({**CFG, "params": {"size": 2}})


def test_anomalous_cell_ranks_first(tmp_path):
    odd = write_cohort(tmp_path)
    run_experiment(ExperimentConfig.from_mapping(CFG), tmp_path, tmp_path / "out")
    rows = list(csv.DictReader((tmp_path / "out" / "scores.csv").open()))
    assert rows[0]["scan_id"] == odd
    assert len(rows) == 17
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m["n_scans_scored"] == 17 and m["n_scans_dropped"] == 0
    assert m["image_slots"] == ["axial[0]", "axial[1]", "radial[0]", "radial[1]"]
    for name in ("top_outliers.png", "most_typical.png"):
        with Image.open(tmp_path / "out" / name) as im:
            assert im.width > 0


def test_images_found_when_nested(tmp_path):
    write_cohort(tmp_path, n_normal=3, nested=True)
    refs, meta = load_cohort(tmp_path)
    assert all(r.path.is_file() for r in refs)
    assert meta["1000"]["cell_serial_number"] == "SN0"


def test_data_behind_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    write_cohort(real, n_normal=3)
    mount = tmp_path / "input"
    mount.mkdir()
    (mount / "batteryct-x").symlink_to(real, target_is_directory=True)
    refs, _ = load_cohort(mount)
    assert len(refs) == 4 * 4 and all(r.path.is_file() for r in refs)


def test_incomplete_scan_is_dropped(tmp_path):
    write_cohort(tmp_path, n_normal=4)
    refs, _ = load_cohort(tmp_path)
    refs = [r for r in refs if not (r.scan_id == "1000" and r.view == "radial")]
    fm = build_features(refs, ExperimentConfig.from_mapping(CFG))
    assert "1000" in fm.dropped and "1000" not in fm.scan_ids


def test_pca_scores_flag_planted_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 30))
    X[7] += 6.0
    s = pca_residual_scores(X, 3, [("all", 0, 30)])
    assert int(np.argmax(s.score)) == 7
    assert s.block_residual.shape == (40, 1)


def write_radial_only_cohort(root: Path, n_normal: int = 16) -> str:
    """Normal cells plus one whose ONLY difference is a void in the radial views."""
    rows = []
    for i in range(n_normal + 1):
        sid = f"{2000 + i}"
        odd = i == n_normal
        for pos in (0.0, 45.0, 90.0, 135.0):
            rel = f"images/{sid}/axial_{int(pos)}.png"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            save_image(axial_image(i * 7 + int(pos)), root / rel)
            rows.append([sid, "axial", pos, rel, "true" if i else "false", f"SN{i}"])
        for k, pos in enumerate((16.0, 32.0, 48.0)):
            rel = f"images/{sid}/radial_{int(pos)}.png"
            img = make_cell(96, defect="void" if odd else "none", seed=i * 10 + k).image
            save_image(img, root / rel)
            rows.append([sid, "radial", pos, rel, "true" if i else "false", f"SN{i}"])
    with (root / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    return f"{2000 + n_normal}"


def test_balanced_views_attribute_radial_defect(tmp_path):
    odd = write_radial_only_cohort(tmp_path)
    run_experiment(ExperimentConfig.from_mapping(CFG), tmp_path, tmp_path / "out")
    rows = list(csv.DictReader((tmp_path / "out" / "scores.csv").open()))
    top = rows[0]
    assert top["scan_id"] == odd
    assert top["driving_view"].startswith("radial")
    assert float(top["radial_score"]) > float(top["axial_score"])
    m = json.loads((tmp_path / "out" / "metrics.json").read_text())
    assert m["driving_view_counts"]["radial"] >= 1


def test_view_weights_equalize_view_types():
    blocks = [("axial[0]", 0, 300), ("axial[1]", 300, 600), ("radial[0]", 600, 700)]
    w = view_weights(blocks, 700)
    assert np.isclose((w[:600] ** 2).sum(), (w[600:] ** 2).sum())


def test_review_pack_written(tmp_path):
    write_cohort(tmp_path, n_normal=6)
    cfg = ExperimentConfig.from_mapping({**CFG, "params": {**CFG["params"], "review_top": 3}})
    run_experiment(cfg, tmp_path, tmp_path / "out")
    review = tmp_path / "out" / "review"
    pages = sorted(p.name for p in review.glob("*.jpg"))
    assert len(pages) == 3 and pages[0].startswith("01_")
    index = (review / "index.md").read_text()
    assert all(p in index for p in pages) and "CC BY-NC-SA" in index
    with Image.open(review / pages[0]) as im:
        assert im.mode == "RGB" and im.width >= 4 * 150


def test_exclude_not_in_inventory(tmp_path):
    write_radial_only_cohort(tmp_path, n_normal=5)  # scan 2000 is marked not-in-inventory
    params = {**CFG["params"], "exclude_not_in_inventory": True, "review_top": 0}
    run_experiment(
        ExperimentConfig.from_mapping({**CFG, "params": params}), tmp_path, tmp_path / "o"
    )
    rows = list(csv.DictReader((tmp_path / "o" / "scores.csv").open()))
    assert "2000" not in {r["scan_id"] for r in rows} and len(rows) == 5
    m = json.loads((tmp_path / "o" / "metrics.json").read_text())
    assert m["excluded"] == ["2000"] and not (tmp_path / "o" / "review").exists()
