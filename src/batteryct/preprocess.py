"""Intensity normalisation and cell-centred cropping."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatImage = npt.NDArray[np.float32]


def normalize_percentile(
    image: npt.NDArray[np.floating], low: float = 1.0, high: float = 99.0
) -> FloatImage:
    """Rescale intensities so the given percentiles map to 0 and 1, then clip.

    Percentiles rather than min/max keep single bright metal-streak pixels from
    compressing the rest of the range.
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("require 0 <= low < high <= 100")
    lo, hi = np.percentile(image, [low, high])
    if hi - lo < 1e-12:
        return np.zeros(image.shape, dtype=np.float32)
    out: FloatImage = np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out


def foreground_centroid(image: npt.NDArray[np.floating], threshold: float = 0.2) -> tuple[int, int]:
    """Return the (row, col) centroid of pixels brighter than ``threshold``.

    Falls back to the image centre when nothing exceeds the threshold.
    """
    mask = image > threshold
    if not mask.any():
        return image.shape[0] // 2, image.shape[1] // 2
    rows, cols = np.nonzero(mask)
    return round(float(rows.mean())), round(float(cols.mean()))


def center_crop(
    image: npt.NDArray[np.floating], size: int, center: tuple[int, int] | None = None
) -> FloatImage:
    """Crop a ``size`` x ``size`` square around ``center``, zero-padding at edges."""
    if size <= 0:
        raise ValueError("size must be positive")
    if center is None:
        center = (image.shape[0] // 2, image.shape[1] // 2)
    half = size // 2
    r0, c0 = center[0] - half, center[1] - half
    out = np.zeros((size, size), dtype=np.float32)
    src_r0, src_c0 = max(r0, 0), max(c0, 0)
    src_r1 = min(r0 + size, image.shape[0])
    src_c1 = min(c0 + size, image.shape[1])
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[src_r0 - r0 : src_r1 - r0, src_c0 - c0 : src_c1 - c0] = image[
            src_r0:src_r1, src_c0:src_c1
        ]
    return out
