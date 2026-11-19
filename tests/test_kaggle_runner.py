"""Tests for the Kaggle runner helpers (no Kaggle API calls)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "kaggle_runner" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


make_kernel = load("make_kernel")
collect = load("collect")


def test_kernel_slug():
    assert make_kernel.kernel_slug("li18650-pca") == "batteryct-li18650-pca"
    assert make_kernel.kernel_slug("A b__c") == "batteryct-a-b-c"
    assert len(make_kernel.kernel_slug("x" * 99)) <= 50


def test_build_kernel_for_every_config(tmp_path):
    for cfg in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        out = make_kernel.build(cfg, "someone", "a" * 40, "run1", tmp_path / cfg.stem)
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert meta["id"].startswith("someone/batteryct-")
        assert meta["is_private"] is True and meta["enable_internet"] is True
        assert meta["dataset_sources"][0].startswith("someone/batteryct-")
        run_py = (out / "run.py").read_text()
        compile(run_py, "run.py", "exec")  # generated script must be valid Python
        assert "a" * 40 in run_py and "'run1'" in run_py


def test_build_rejects_bad_input(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: x\n")
    with pytest.raises(SystemExit):
        make_kernel.build(cfg, "u", "a" * 40, "r", tmp_path / "k")
    cfg.write_text("name: x\ndataset: d\n")
    with pytest.raises(SystemExit):
        make_kernel.build(cfg, "u", "abc", "r", tmp_path / "k")


@pytest.mark.parametrize(
    ("text", "state"),
    [
        ('u/k has status "complete"', "complete"),
        ('u/k has status "KernelWorkerStatus.COMPLETE"', "complete"),
        ('u/k has status "KernelWorkerStatus.RUNNING"', "running"),
        ('u/k has status "queued"', "queued"),
        ('u/k has status "error"', "error"),
        ('u/k has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"', "error"),
        ("something odd", "missing"),
    ],
)
def test_parse_status(text, state):
    assert collect.parse_status(text) == state


def test_collect_outputs_copies_small_files(tmp_path):
    src = tmp_path / "src" / "20261119-0000_x_abc1234"
    src.mkdir(parents=True)
    (src / "run_meta.json").write_text(json.dumps({"run_id": "R1", "status": "complete"}))
    (src / "scores.csv").write_text("rank,scan_id\n1,5\n")
    (src / "huge.bin").write_bytes(b"0" * 10)
    (tmp_path / "src" / "kernel.log").write_text("log line\n")
    run_id, dest = collect.collect_outputs(tmp_path / "src", tmp_path / "results")
    assert run_id == "R1"
    assert sorted(p.name for p in dest.iterdir()) == [
        "kaggle_log_tail.txt",
        "run_meta.json",
        "scores.csv",
    ]


def test_collect_outputs_requires_meta(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect.collect_outputs(tmp_path, tmp_path / "out")
