"""SYNTHETIC axial cross-sections of cylindrical cells, for tests and CI only.

These images are procedurally generated. They are NOT derived from the Glimpse
dataset and must never be reported as real CT results. They give CI something
shaped like a cylindrical-cell slice (can wall, spiral jelly roll, central
mandrel hole) so the pipeline can be exercised without downloading real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

FloatImage = npt.NDArray[np.float32]
DefectKind = Literal["none", "void", "streak", "offset_core"]
DEFECT_KINDS: tuple[DefectKind, ...] = ("none", "void", "streak", "offset_core")


@dataclass(frozen=True)
class SyntheticCell:
    """One synthetic slice and its ground truth."""

    image: FloatImage
    defect: DefectKind
    seed: int


def make_cell(
    size: int = 128,
    defect: DefectKind = "none",
    seed: int = 0,
    noise: float = 0.03,
) -> SyntheticCell:
    """Generate one synthetic axial slice of a cylindrical cell.

    Defects:
        ``void``: a dark gas-pocket-like blob inside the jelly roll.
        ``streak``: a bright line through the slice (stands in for a metal-streak
            scan artifact, which is NOT a cell defect).
        ``offset_core``: the winding centre is displaced from the can centre.
    """
    if defect not in DEFECT_KINDS:
        raise ValueError(f"unknown defect {defect!r}; choose from {DEFECT_KINDS}")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    c = (size - 1) / 2.0
    can_r = 0.45 * size
    core_r = 0.06 * size

    # Normal manufacturing variation: small jitter in winding centre and pitch.
    jitter = rng.normal(0.0, 0.004 * size, 2)
    cy, cx = c + jitter[0], c + jitter[1]
    if defect == "offset_core":
        angle = rng.uniform(0, 2 * np.pi)
        cy += 0.06 * size * np.sin(angle)
        cx += 0.06 * size * np.cos(angle)
    pitch = 0.035 * size * rng.uniform(0.97, 1.03)

    r_can = np.hypot(yy - c, xx - c)
    r_core = np.hypot(yy - cy, xx - cx)
    theta = np.arctan2(yy - cy, xx - cx)

    img = np.zeros((size, size), dtype=np.float32)
    # Archimedean spiral: brightness oscillates with (r - pitch * theta / 2pi).
    phase = 2 * np.pi * (r_core - pitch * theta / (2 * np.pi)) / pitch
    roll = 0.45 + 0.2 * np.cos(phase)
    inside_roll = (r_can < can_r - 2) & (r_core > core_r)
    img[inside_roll] = roll[inside_roll]
    img[np.abs(r_can - can_r) < 2.0] = 0.95  # steel can wall

    if defect == "void":
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0.35, 0.7) * can_r
        vy, vx = c + rad * np.sin(ang), c + rad * np.cos(ang)
        vr = rng.uniform(0.04, 0.07) * size
        img[np.hypot(yy - vy, xx - vx) < vr] = 0.05
    elif defect == "streak":
        ang = rng.uniform(0, np.pi)
        dist = np.abs((yy - c) * np.cos(ang) - (xx - c) * np.sin(ang))
        img += (0.35 * np.exp(-(dist**2) / 2.0)).astype(np.float32)

    img += rng.normal(0.0, noise, img.shape).astype(np.float32)
    return SyntheticCell(np.clip(img, 0.0, 1.0).astype(np.float32), defect, seed)


def make_cohort(
    n_normal: int, n_per_defect: int = 0, size: int = 128, seed: int = 0
) -> list[SyntheticCell]:
    """Generate a cohort of normal cells plus ``n_per_defect`` of each defect kind."""
    cells: list[SyntheticCell] = []
    s = seed
    plan: list[DefectKind] = ["none"] * n_normal
    for kind in DEFECT_KINDS[1:]:
        plan += [kind] * n_per_defect
    for kind in plan:
        cells.append(make_cell(size=size, defect=kind, seed=s))
        s += 1
    return cells
