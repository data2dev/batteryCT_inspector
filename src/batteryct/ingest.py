"""Select and download CT views from the Glimpse dataset on Figshare.

Archive layout (checked on the sodium-ion 18650 archive, 2026-11-11)::

    <top>/slices/<scan_id>/axial_images/<angle_deg>.png    # 36 per scan, 0-175 deg
    <top>/slices/<scan_id>/radial_images/<height>.png      # ~138 per scan, along the axis

"Axial" images are longitudinal slices through the cell axis at a rotation angle;
"radial" images are cross-sections at a height along the axis (units presumed mm,
not stated in the archive). Only the selected views are fetched, using HTTP range
requests (see :mod:`batteryct.remote_zip`).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from batteryct import __version__
from batteryct.remote_zip import USER_AGENT, RemoteZip

FIGSHARE_ARTICLE_ID = 25330501
FIGSHARE_VERSION = 1
FIGSHARE_DOI = "10.25452/figshare.plus.25330501"
DATA_PAPER_DOI = "10.1016/j.dib.2024.110614"
INVENTORY_FILE_ID = 44816284


def figshare_url(file_id: int) -> str:
    return f"https://ndownloader.figshare.com/files/{file_id}"


@dataclass(frozen=True)
class Cohort:
    key: str
    file_id: int
    batch: str  # value of the "Batch" column in Scan_inventory.csv
    note: str = ""


# Accessibility as checked from a cloud host on 2026-11-11. Three archives return
# HTTP 403 (explicit deny on Figshare's temporary storage bucket).
COHORTS: dict[str, Cohort] = {
    c.key: c
    for c in [
        Cohort("na18650", 44809771, "Demo: 18650 cylindrical (sodium ion)"),
        Cohort("he21700", 44815270, "Demo: 2170 cylindrical (high energy density)"),
        Cohort("pouch", 44810215, "Demo: Pouch"),
        Cohort("prismatic", 44809954, "Demo: Prismatic"),
        Cohort("li18650", 44813026, "Demo: 18650 cylindrical", "HTTP 403 as of 2026-11-11"),
        Cohort("li21700", 44816857, "Demo: 2170 cylindrical", "HTTP 403 as of 2026-11-11"),
        Cohort("c4680", 44815120, "Demo: 4680 cylindrical", "HTTP 403 as of 2026-11-11"),
    ]
}

_MEMBER_RE = re.compile(
    r"(?:^|/)slices/(?P<scan>[^/]+)/(?P<view>\w+?)_images/(?P<pos>[-\d.eE]+)\.png$"
)


@dataclass(frozen=True)
class ViewSelection:
    """Which views to keep per scan.

    ``axial_angles_deg``: nearest available axial slice to each angle.
    ``radial_fractions``: positions as fractions of each scan's radial range
    (0 = first slice, 1 = last), so they adapt to cell length.
    """

    axial_angles_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    radial_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class IngestConfig:
    cohort: str
    views: ViewSelection = field(default_factory=ViewSelection)
    limit_scans: int | None = None

    @classmethod
    def from_mapping(cls, d: Mapping[str, Any]) -> IngestConfig:
        if d.get("cohort") not in COHORTS:
            raise ValueError(f"cohort must be one of {sorted(COHORTS)}, got {d.get('cohort')!r}")
        v = d.get("views", {}) or {}
        defaults = ViewSelection()
        views = ViewSelection(
            tuple(float(a) for a in v.get("axial_angles_deg", defaults.axial_angles_deg)),
            tuple(float(f) for f in v.get("radial_fractions", defaults.radial_fractions)),
        )
        for f in views.radial_fractions:
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"radial fraction {f} outside [0, 1]")
        limit = d.get("limit_scans")
        return cls(d["cohort"], views, int(limit) if limit is not None else None)


@dataclass(frozen=True)
class Slice:
    member: str
    scan_id: str
    view: str
    position: float


def parse_members(names: Iterable[str]) -> dict[str, dict[str, list[Slice]]]:
    """Group archive members into ``{scan_id: {view: [Slice, ...] sorted by position}}``."""
    scans: dict[str, dict[str, list[Slice]]] = defaultdict(lambda: defaultdict(list))
    for name in names:
        m = _MEMBER_RE.search(name)
        if m:
            s = Slice(name, m["scan"], m["view"], float(m["pos"]))
            scans[s.scan_id][s.view].append(s)
    for views in scans.values():
        for lst in views.values():
            lst.sort(key=lambda s: s.position)
    return {k: dict(v) for k, v in scans.items()}


def _nearest(slices: list[Slice], target: float) -> Slice:
    return min(slices, key=lambda s: (abs(s.position - target), s.position))


def select_views(views: Mapping[str, list[Slice]], sel: ViewSelection) -> list[Slice]:
    """Pick the configured views for one scan (deduplicated, stable order)."""
    chosen: list[Slice] = []
    axial = views.get("axial", [])
    if axial:
        chosen += [_nearest(axial, a % 180.0) for a in sel.axial_angles_deg]
    radial = views.get("radial", [])
    if radial:
        lo, hi = radial[0].position, radial[-1].position
        chosen += [_nearest(radial, lo + f * (hi - lo)) for f in sel.radial_fractions]
    unique: dict[str, Slice] = {}
    for s in chosen:
        unique.setdefault(s.member, s)
    return list(unique.values())


def load_inventory(source: str | Path | None = None) -> list[dict[str, str]]:
    """Read Scan_inventory.csv from a local path, or download it from Figshare."""
    if source is not None and Path(source).exists():
        text = Path(source).read_text(encoding="utf-8-sig")
    else:
        url = str(source) if source else figshare_url(INVENTORY_FILE_ID)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = resp.read().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


MANIFEST_FIELDS = [
    "cohort",
    "scan_id",
    "in_inventory",
    "cell_serial_number",
    "cell_model_name",
    "batch_id",
    "view",
    "position",
    "file",
    "zip_member",
    "width",
    "height",
    "bytes",
    "sha256",
]


def _slug(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def run_ingest(
    config: IngestConfig,
    out_dir: str | Path,
    archive: RemoteZip | None = None,
    inventory: list[dict[str, str]] | None = None,
    workers: int = 8,
    log: bool = True,
) -> Path:
    """Fetch the selected views into ``out_dir`` and write ``manifest.csv``.

    Re-running into the same directory skips files that already exist.
    Returns the manifest path.
    """
    cohort = COHORTS[config.cohort]
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    if archive is None:
        archive = RemoteZip(figshare_url(cohort.file_id))
    if inventory is None:
        inventory = load_inventory()
    inv = {r["scan_id"]: r for r in inventory if r.get("Batch") == cohort.batch}

    scans = parse_members(archive.names())
    scan_ids = sorted(scans, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s))
    if config.limit_scans is not None:
        scan_ids = scan_ids[: config.limit_scans]
    jobs = [s for sid in scan_ids for s in select_views(scans[sid], config.views)]
    if log:
        print(f"[ingest] {cohort.key}: {len(scan_ids)} scans, {len(jobs)} images to fetch")

    def fetch_one(s: Slice) -> dict[str, str]:
        rel = Path("images") / s.scan_id / f"{s.view}_{_slug(s.position)}.png"
        dest = out / rel
        if dest.exists():
            data = dest.read_bytes()
        else:
            data = archive.read(s.member)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
        meta = inv.get(s.scan_id, {})
        return {
            "cohort": cohort.key,
            "scan_id": s.scan_id,
            "in_inventory": str(bool(meta)).lower(),
            "cell_serial_number": meta.get("cell_serial_number", ""),
            "cell_model_name": meta.get("cell_model_name", ""),
            "batch_id": meta.get("batch_id", ""),
            "view": s.view,
            "position": f"{s.position:g}",
            "file": rel.as_posix(),
            "zip_member": s.member,
            "width": str(w),
            "height": str(h),
            "bytes": str(len(data)),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, row in enumerate(pool.map(fetch_one, jobs), 1):
            rows.append(row)
            if log and (i % 50 == 0 or i == len(jobs)):
                print(f"[ingest] {i}/{len(jobs)}")

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    meta_out = {
        "cohort": asdict(cohort),
        "config": asdict(config),
        "source": {
            "figshare_article_id": FIGSHARE_ARTICLE_ID,
            "figshare_version": FIGSHARE_VERSION,
            "figshare_doi": FIGSHARE_DOI,
            "data_paper_doi": DATA_PAPER_DOI,
            "archive_url": archive.url,
            "archive_bytes": archive.size,
            "archive_members": len(archive.members),
        },
        "counts": {
            "scans_in_archive": len(scans),
            "scans_selected": len(scan_ids),
            "scans_not_in_inventory": sum(sid not in inv for sid in scan_ids),
            "images": len(rows),
            "bytes": sum(int(r["bytes"]) for r in rows),
        },
        "batteryct_version": __version__,
        "git_sha": os.environ.get("GITHUB_SHA", ""),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "ingest_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    (out / "README.md").write_text(_DATA_README.format(cohort=cohort.key))
    return manifest


def write_kaggle_metadata(out_dir: str | Path, dataset_id: str, title: str) -> Path:
    """Write ``dataset-metadata.json`` for ``kaggle datasets create/version``."""
    if "/" not in dataset_id:
        raise ValueError("dataset_id must look like '<kaggle-username>/<slug>'")
    path = Path(out_dir) / "dataset-metadata.json"
    path.write_text(
        json.dumps(
            {
                "title": title[:50],
                "id": dataset_id,
                "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
                "subtitle": "Selected views from the Glimpse battery CT dataset (modified subset)"[
                    :80
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return path


_DATA_README = """# BatteryCT-Inspector working subset: {cohort}

Selected CT views extracted from:

> Condon, A. et al. A dataset of over one thousand computed tomography scans of battery cells.
> Data in Brief 55 (2024) 110614. https://doi.org/10.1016/j.dib.2024.110614
> Dataset (Glimpse): https://doi.org/10.25452/figshare.plus.25330501

License: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/).
Non-commercial use only; adaptations must use the same license.

This is a subset: only the views listed in `manifest.csv` were extracted. Files are
byte-identical to the archive members (CRC-checked); file names were changed.
Provenance is recorded in `ingest_meta.json`.
"""
