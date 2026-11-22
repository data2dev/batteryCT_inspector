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


def fake_wheel(tmp_path: Path) -> Path:
    w = tmp_path / "batteryct_inspector-0.0.1-py3-none-any.whl"
    w.write_bytes(b"PK\x05\x06" + b"\0" * 18)  # empty zip; content irrelevant here
    return w


def test_build_kernel_for_every_config(tmp_path):
    wheel = fake_wheel(tmp_path)
    for cfg in sorted((ROOT / "configs" / "experiments").glob("*.yaml")):
        out = make_kernel.build(cfg, "someone", "a" * 40, "run1", wheel, tmp_path / cfg.stem)
        meta = json.loads((out / "kernel-metadata.json").read_text())
        assert meta["id"].startswith("someone/batteryct-")
        assert meta["is_private"] is True and meta["enable_internet"] is False
        assert meta["dataset_sources"][0].startswith("someone/batteryct-")
        run_py = (out / "run.py").read_text()
        compile(run_py, "run.py", "exec")  # generated script must be valid Python
        assert "a" * 40 in run_py and "'run1'" in run_py


def test_build_rejects_bad_input(tmp_path):
    wheel = fake_wheel(tmp_path)
    cfg = tmp_path / "c.yaml"
    cfg.write_text("name: x\n")
    with pytest.raises(SystemExit):
        make_kernel.build(cfg, "u", "a" * 40, "r", wheel, tmp_path / "k")
    cfg.write_text("name: x\ndataset: d\n")
    with pytest.raises(SystemExit):
        make_kernel.build(cfg, "u", "abc", "r", wheel, tmp_path / "k")
    with pytest.raises(SystemExit):
        make_kernel.build(cfg, "u", "a" * 40, "r", tmp_path / "missing.whl", tmp_path / "k")
    cfg.write_text("name: x\ndataset: d\ninternet: true\n")
    out = make_kernel.build(cfg, "u", "a" * 40, "r", wheel, tmp_path / "k")
    assert json.loads((out / "kernel-metadata.json").read_text())["enable_internet"] is True


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


def test_gpu_config_requests_t4(tmp_path):
    cfg = tmp_path / "g.yaml"
    cfg.write_text("name: x\ndataset: d\ngpu: true\n")
    out = make_kernel.build(cfg, "u", "a" * 40, "r", fake_wheel(tmp_path), tmp_path / "k")
    meta = json.loads((out / "kernel-metadata.json").read_text())
    assert meta["enable_gpu"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    cfg.write_text("name: x\ndataset: d\n")
    out = make_kernel.build(cfg, "u", "a" * 40, "r", fake_wheel(tmp_path), tmp_path / "k2")
    assert "machine_shape" not in json.loads((out / "kernel-metadata.json").read_text())


summarize_run = load("summarize_run")


def test_summary_handles_pca_and_patchcore(tmp_path):
    pca = tmp_path / "pca"
    pca.mkdir()
    (pca / "run_meta.json").write_text(json.dumps({"status": "complete", "git_sha": "abcdef12"}))
    (pca / "scores.csv").write_text("rank,scan_id,score,driving_view\n1,5,3.2,axial[0]\n")
    assert "axial[0]" in summarize_run.summarize(pca)

    pc = tmp_path / "pc"
    (pc / "review").mkdir(parents=True)
    (pc / "review" / "index.md").write_text("x")
    (pc / "run_meta.json").write_text(json.dumps({"status": "complete"}))
    (pc / "metrics.json").write_text(
        json.dumps(
            {
                "method": "patchcore",
                "n_cells_scored": 40,
                "planted": {
                    "n": 12,
                    "cell_detected_p95": 0.67,
                    "located": 0.83,
                    "by_type": {"void": {"cell_detected_p95": 0.5, "located": 0.75}},
                },
            }
        )
    )
    (pc / "scores.csv").write_text("rank,scan_id,score,hotspot_view\n1,1794,1.15,radial[0]\n")
    text = summarize_run.summarize(pc)
    assert "radial[0]" in text and "planted anomalies**: 12" in text and "review/index.md" in text


def test_summary_survives_empty_dir(tmp_path):
    assert "status" in summarize_run.summarize(tmp_path)
