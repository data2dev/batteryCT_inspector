# Data: source, layout, access, ingest

## Source

Glimpse battery CT dataset, Figshare+ article 25330501 (version 1), CC BY-NC-SA 4.0.
Paper: Condon et al., Data in Brief 55 (2024) 110614, https://doi.org/10.1016/j.dib.2024.110614.

## Archives and access (checked 2026-11-11)

| Key | Archive | Cells (inventory) | Scan folders | Size |
|---|---|---|---|---|
| `li18650` | Demo_ 18650 cylindrical.zip | 400 | 406 | 91.3 GB |
| `li21700` | Demo_ 2170 cylindrical.zip | 500 | 506 | 122.5 GB |
| `na18650` | Demo_ 18650 cylindrical (sodium ion).zip | 50 | 55 | 11.1 GB |
| `he21700` | Demo_ 2170 cylindrical (high energy density).zip | 25 | | 6.3 GB |
| `c4680` | Demo_ 4680 cylindrical.zip | 25 | 26 | 4.8 GB |
| `pouch` | Demo_ Pouch.zip | 10 | | 2.1 GB |
| `prismatic` | Demo_ Prismatic.zip | 5 | | 2.4 GB |

**Network-dependent access.** `li18650`, `li21700` and `c4680` (the files with no MD5
in the Figshare API) redirect to a different storage bucket that answers
`AccessDenied ... explicit deny in a resource-based policy` to requests from some
cloud networks, but serves GitHub-hosted runners normally (verified 2026-11-11 by
ingesting 2 scans from each). If you get HTTP 403, run the **Ingest cohort**
workflow instead of ingesting locally.

## Archive layout

```
<top>/slices/<scan_id>/axial_images/<angle>.png     36 per scan, 0-175 deg in 5 deg steps
<top>/slices/<scan_id>/radial_images/<height>.png   ~138 per scan, -0.5 .. ~65.4 (presumably mm)
```

- **axial**: longitudinal slice through the cell axis at a rotation angle. Size varies by
  cohort and scan: 1320 x 4584/4612 (18650), 1340 x 4328 (21700), 1342 x 2326 (4680).
- **radial**: cross-section perpendicular to the axis at a height. 1320 x 1320 px.
  Positions near the ends show the caps; mid-height shows the jelly-roll spiral.
- 8-bit grayscale PNG with a narrow intensity range (e.g. max ~115).
- `scan_id` matches `scan_id` in `Scan_inventory.csv`. In `na18650`, 5 of 55 scan
  folders (2194, 2208, 2209, 2210, 2220) are **not** in the inventory; `li18650`,
  `li21700` and `c4680` also have more scan folders than inventory rows. These may be
  rescans; check before splitting by cell.

## Ingest

`batteryct ingest configs/ingest/<cohort>.yaml OUT` reads the archive's central
directory with HTTP range requests, then fetches only the selected views (one range
request per image, CRC-checked). Default selection: axial at 0/45/90/135 deg, radial
at 25/50/75 % of the height range, i.e. 7 images per scan.

`na18650` full run: 55 scans, 385 images, 0.70 GB, ~1 minute.

Output: `images/<scan_id>/<view>_<position>.png`, `manifest.csv` (sha256, size, source
member, inventory join), `ingest_meta.json` (provenance), `README.md` (attribution).

The GitHub workflow **Ingest cohort** runs this on a GitHub runner, uploads the result to
a private Kaggle dataset `<kaggle-user>/batteryct-<cohort>`, and opens a PR adding the
manifest under `results/manifests/<cohort>/`.
