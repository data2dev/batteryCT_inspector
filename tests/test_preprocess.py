import numpy as np
import pytest

from batteryct.preprocess import center_crop, foreground_centroid, normalize_percentile
from batteryct.synthetic import make_cell


def test_normalize_percentile_ignores_hot_pixel():
    img = np.linspace(0, 1, 10_000, dtype=np.float32).reshape(100, 100)
    img[0, 0] = 1000.0  # metal-streak-like outlier
    out = normalize_percentile(img)
    assert out.dtype == np.float32
    assert out.min() == 0.0 and out.max() == 1.0
    assert 0.4 < float(np.median(out)) < 0.6


def test_normalize_constant_image():
    out = normalize_percentile(np.full((8, 8), 3.0))
    assert np.all(out == 0.0)


def test_normalize_bad_percentiles():
    with pytest.raises(ValueError):
        normalize_percentile(np.zeros((4, 4)), low=50, high=10)


def test_centroid_of_synthetic_cell_near_centre():
    img = make_cell(size=96, seed=0).image
    r, c = foreground_centroid(img)
    assert abs(r - 48) <= 3 and abs(c - 48) <= 3


def test_centroid_empty_falls_back_to_centre():
    assert foreground_centroid(np.zeros((10, 20))) == (5, 10)


def test_center_crop_inside():
    img = np.arange(100, dtype=np.float32).reshape(10, 10)
    out = center_crop(img, 4, center=(5, 5))
    np.testing.assert_array_equal(out, img[3:7, 3:7])


def test_center_crop_pads_at_edge():
    img = np.ones((10, 10), dtype=np.float32)
    out = center_crop(img, 6, center=(0, 0))
    assert out.shape == (6, 6)
    assert out[:3, :].sum() == 0 and out[:, :3].sum() == 0
    assert out[3:, 3:].sum() == 9


def test_center_crop_rejects_bad_size():
    with pytest.raises(ValueError):
        center_crop(np.zeros((4, 4)), 0)
