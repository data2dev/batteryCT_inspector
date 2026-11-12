"""Ingest tests on a SYNTHETIC archive that mimics the Glimpse layout (no network)."""

import csv
import io
import json

import pytest
from PIL import Image

from batteryct.ingest import (
    COHORTS,
    IngestConfig,
    ViewSelection,
    parse_members,
    run_ingest,
    select_views,
    write_kaggle_metadata,
)
from batteryct.remote_zip import RemoteZip
from tests.test_remote_zip import URL, FakeServer, make_zip


def png(w: int, h: int, v: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (w, h), v).save(buf, format="PNG")
    return buf.getvalue()


def synthetic_archive(scan_ids: tuple[str, ...] = ("101", "102", "103")) -> dict[str, bytes]:
    files = {}
    for k, sid in enumerate(scan_ids):
        for ang in range(0, 180, 5):
            files[f"Na-ion/slices/{sid}/axial_images/{float(ang)}.png"] = png(8, 24, k)
        for i in range(-1, 131):
            files[f"Na-ion/slices/{sid}/radial_images/{i * 0.5}.png"] = png(8, 8, k)
    return files


INVENTORY = [
    {
        "Batch": COHORTS["na18650"].batch,
        "batch_id": "1117",
        "cell_model_name": "SIB18650/3V",
        "cell_serial_number": f"Na Ion {n}",
        "scan_id": sid,
    }
    for n, sid in enumerate(["101", "102"])
] + [{"Batch": "Demo: Pouch", "scan_id": "103", "cell_serial_number": "wrong batch"}]


def test_parse_members_groups_and_sorts():
    scans = parse_members([*synthetic_archive(("7",)), "Na-ion/readme.txt"])
    assert list(scans) == ["7"]
    assert len(scans["7"]["axial"]) == 36
    radial = scans["7"]["radial"]
    assert radial[0].position == -0.5 and radial[-1].position == 65.0
    assert [s.position for s in radial] == sorted(s.position for s in radial)


def test_select_views_nearest_and_dedup():
    scans = parse_members(synthetic_archive(("7",)))
    sel = ViewSelection(axial_angles_deg=(0, 47, 180), radial_fractions=(0.0, 0.5, 0.5, 1.0))
    got = [(s.view, s.position) for s in select_views(scans["7"], sel)]
    # 180 deg wraps to 0 and is de-duplicated; 47 -> 45; 0.5 of [-0.5, 65] -> 32.0 or 32.5
    assert got[:2] == [("axial", 0.0), ("axial", 45.0)]
    radial = [p for v, p in got if v == "radial"]
    assert radial[0] == -0.5 and radial[-1] == 65.0 and len(radial) == 3
    assert abs(radial[1] - 32.25) <= 0.25


def test_config_validation():
    assert IngestConfig.from_mapping({"cohort": "na18650"}).views == ViewSelection()
    with pytest.raises(ValueError):
        IngestConfig.from_mapping({"cohort": "nope"})
    with pytest.raises(ValueError):
        IngestConfig.from_mapping({"cohort": "na18650", "views": {"radial_fractions": [1.5]}})


def test_run_ingest_end_to_end(tmp_path):
    blob = make_zip(synthetic_archive())
    rz = RemoteZip(URL, size=len(blob), fetch=FakeServer(blob))
    cfg = IngestConfig("na18650", ViewSelection((0, 90), (0.5,)), limit_scans=2)
    manifest = run_ingest(cfg, tmp_path, archive=rz, inventory=INVENTORY, workers=2, log=False)

    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 2 * 3
    assert {r["scan_id"] for r in rows} == {"101", "102"}
    assert all(r["in_inventory"] == "true" for r in rows)
    assert rows[0]["cell_serial_number"] == "Na Ion 0"
    for r in rows:
        assert (tmp_path / r["file"]).exists()
        assert len(r["sha256"]) == 64
    ax = next(r for r in rows if r["view"] == "axial")
    assert (ax["width"], ax["height"]) == ("8", "24")

    meta = json.loads((tmp_path / "ingest_meta.json").read_text())
    assert meta["counts"]["scans_in_archive"] == 3
    assert meta["counts"]["images"] == 6
    assert "CC BY-NC-SA 4.0" in (tmp_path / "README.md").read_text()

    # Re-run is idempotent and does not refetch existing files.
    server = FakeServer(blob)
    rz.fetch = server
    run_ingest(cfg, tmp_path, archive=rz, inventory=INVENTORY, log=False)
    assert server.calls == 0


def test_scan_missing_from_inventory(tmp_path):
    blob = make_zip(synthetic_archive(("103",)))
    rz = RemoteZip(URL, size=len(blob), fetch=FakeServer(blob))
    run_ingest(IngestConfig("na18650"), tmp_path, archive=rz, inventory=INVENTORY, log=False)
    rows = list(csv.DictReader((tmp_path / "manifest.csv").open()))
    # scan 103 exists in the inventory only under a different batch -> not matched
    assert rows and all(r["in_inventory"] == "false" for r in rows)


def test_kaggle_metadata(tmp_path):
    p = write_kaggle_metadata(tmp_path, "someone/batteryct-na18650", "batteryct na18650 views")
    meta = json.loads(p.read_text())
    assert meta["licenses"] == [{"name": "CC-BY-NC-SA-4.0"}]
    with pytest.raises(ValueError):
        write_kaggle_metadata(tmp_path, "no-slash", "t")
