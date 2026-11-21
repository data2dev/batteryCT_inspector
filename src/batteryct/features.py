"""Pretrained CNN patch features for PatchCore (needs the optional ``ml`` extra: torch).

WideResNet-50-2 (ImageNet weights) layer2 + layer3 activations, 3x3 neighbourhood
averaging, layer3 upsampled to layer2's grid (stride 8), concatenated (1536-d) and
reduced with a fixed, seeded random projection.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt


def find_weights(name: str, search_roots: list[str | Path]) -> Path:
    """Locate a weights file by name (or path) under the given roots, following symlinks."""
    if Path(name).is_file():
        return Path(name)
    for root in search_roots:
        for dirpath, _, files in os.walk(root, followlinks=True):
            if name in files:
                return Path(dirpath) / name
    raise FileNotFoundError(
        f"weights file {name!r} not found under {search_roots}. On Kaggle, attach the "
        "private weights dataset (see .github/workflows/upload-weights.yml)."
    )


def default_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class WrnExtractor:
    """Callable: prepared [0, 1] image (H, W) -> feature grid (H/8*scale, W/8*scale, proj_dim)."""

    def __init__(
        self, weights: Path, device: str, scale: float, proj_dim: int, seed: int = 0
    ) -> None:
        import torch
        import torchvision

        model = torchvision.models.wide_resnet50_2(weights=None)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.eval().to(device)
        self.device = device
        self.scale = scale
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.randn(1536, proj_dim, generator=g) / math.sqrt(proj_dim)).to(device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, img: npt.NDArray[np.float32], view: str) -> npt.NDArray[np.float32]:
        import torch
        import torch.nn.functional as F

        m = self.model
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))[None, None]
            x = x.to(self.device)
            if self.scale != 1.0:
                x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", antialias=True)
            x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            l2 = m.layer2(m.layer1(x))
            l3 = m.layer3(l2)
            f2 = F.avg_pool2d(l2, 3, stride=1, padding=1)
            f3 = F.avg_pool2d(l3, 3, stride=1, padding=1)
            f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            f = torch.cat([f2, f3], dim=1)[0].permute(1, 2, 0) @ self.proj
        out: npt.NDArray[np.float32] = f.float().cpu().numpy()
        return out
