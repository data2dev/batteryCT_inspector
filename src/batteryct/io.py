"""Image loading and saving for CT cross-sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

FloatImage = npt.NDArray[np.float32]


def load_image(path: str | Path) -> FloatImage:
    """Load an image as a 2D float32 array scaled to [0, 1].

    Colour images are converted to luminance. 16-bit images keep their full range.
    """
    with Image.open(path) as img:
        if img.mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            max_val = 65535.0 if img.mode.startswith("I;16") else max(float(arr.max()), 1.0)
        else:
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            max_val = 255.0
    out: FloatImage = np.clip(arr / max_val, 0.0, 1.0).astype(np.float32)
    return out


def save_image(image: npt.NDArray[np.floating], path: str | Path) -> None:
    """Save a 2D array with values in [0, 1] as an 8-bit grayscale PNG."""
    if image.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {image.shape}")
    arr = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0)
    Image.fromarray((arr * 255.0).round().astype(np.uint8), mode="L").save(path)
