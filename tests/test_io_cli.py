import csv

import numpy as np
import pytest
from PIL import Image

from batteryct import __version__
from batteryct.cli import main
from batteryct.io import load_image, save_image


def test_save_load_roundtrip(tmp_path):
    img = np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / "x.png"
    save_image(img, path)
    back = load_image(path)
    assert back.shape == img.shape
    assert np.abs(back - img).max() <= 1 / 255 + 1e-6


def test_load_16bit(tmp_path):
    arr = np.full((4, 4), 65535, dtype=np.uint16)
    path = tmp_path / "x16.png"
    Image.fromarray(arr).save(path)
    assert np.allclose(load_image(path), 1.0)


def test_save_rejects_3d(tmp_path):
    with pytest.raises(ValueError):
        save_image(np.zeros((2, 2, 3)), tmp_path / "bad.png")


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_synth_writes_manifest(tmp_path):
    assert (
        main(["synth", str(tmp_path), "--n-normal", "3", "--n-per-defect", "1", "--size", "32"])
        == 0
    )
    with (tmp_path / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert all(r["synthetic"] == "true" for r in rows)
    for r in rows:
        assert load_image(tmp_path / r["file"]).shape == (32, 32)
