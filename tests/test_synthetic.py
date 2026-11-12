"""Tests for SYNTHETIC fixture generation (no real CT data involved)."""

import numpy as np
import pytest

from batteryct.synthetic import DEFECT_KINDS, make_cell, make_cohort


@pytest.mark.parametrize("defect", DEFECT_KINDS)
def test_make_cell_shape_range_dtype(defect):
    cell = make_cell(size=64, defect=defect, seed=1)
    assert cell.image.shape == (64, 64)
    assert cell.image.dtype == np.float32
    assert cell.image.min() >= 0.0 and cell.image.max() <= 1.0
    assert cell.defect == defect


def test_make_cell_is_deterministic():
    a = make_cell(seed=7, defect="void").image
    b = make_cell(seed=7, defect="void").image
    np.testing.assert_array_equal(a, b)


def test_seeds_differ():
    assert not np.array_equal(make_cell(seed=1).image, make_cell(seed=2).image)


def test_void_darkens_roll():
    normal = make_cell(seed=3, noise=0.0).image
    void = make_cell(seed=3, defect="void", noise=0.0).image
    assert (void < 0.1).sum() > (normal < 0.1).sum()


def test_unknown_defect_rejected():
    with pytest.raises(ValueError):
        make_cell(defect="crack")  # type: ignore[arg-type]


def test_make_cohort_counts():
    cells = make_cohort(n_normal=5, n_per_defect=2)
    assert len(cells) == 5 + 2 * (len(DEFECT_KINDS) - 1)
    assert sum(c.defect == "none" for c in cells) == 5
    assert len({c.seed for c in cells}) == len(cells)
