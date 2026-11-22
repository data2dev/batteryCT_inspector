"""Markdown summary of a collected run directory, for the results PR body.

Works for every method (PCA, PatchCore, ...): it only reports fields that exist.
Never raises on missing or unexpected content; prints what it can.

Usage: python kaggle_runner/summarize_run.py results/runs/<run_id>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def summarize(d: Path, top: int = 10) -> str:
    out: list[str] = []
    meta = _load(d / "run_meta.json")
    out.append(
        f"- **status**: {meta.get('status')}  \n"
        f"- **commit**: `{str(meta.get('git_sha', ''))[:7]}`  \n"
        f"- **runtime**: {meta.get('runtime_s')} s  \n"
        f"- **gpu**: {meta.get('gpu')}"
    )
    m = _load(d / "metrics.json")
    if m:
        n = m.get("n_scans_scored", m.get("n_cells_scored"))
        out.append(f"- **method**: {m.get('method', m.get('config', {}).get('method', '?'))}")
        out.append(f"- **cells scored**: {n}")
        out.append(f"- **score quantiles**: {m.get('score_quantiles')}, max {m.get('score_max')}")
        pl = m.get("planted")
        if isinstance(pl, dict) and pl.get("n"):
            out.append(
                f"- **planted anomalies**: {pl['n']}; flagged (>p95 of real cells) "
                f"{pl.get('cell_detected_p95')}, located {pl.get('located')}"
            )
            for typ, v in (pl.get("by_type") or {}).items():
                out.append(
                    f"  - {typ}: flagged {v.get('cell_detected_p95')}, located {v.get('located')}"
                )
    try:
        rows = list(csv.DictReader((d / "scores.csv").open()))[:top]
    except OSError:
        rows = []
    if rows:
        where = next((c for c in ("driving_view", "hotspot_view") if c in rows[0]), None)
        head = "| rank | scan | score |" + (f" {where.replace('_', ' ')} |" if where else "")
        out.append("\n" + head + "\n|---|---|---|" + ("---|" if where else ""))
        for r in rows:
            cell = f" {r.get(where, '')} |" if where else ""
            out.append(f"| {r.get('rank')} | {r.get('scan_id')} | {r.get('score')} |{cell}")
    if (d / "review" / "index.md").exists():
        out.append(f"\nReview pack: `{d.as_posix()}/review/index.md`")
    return "\n".join(out)


if __name__ == "__main__":
    try:
        print(summarize(Path(sys.argv[1])))
    except Exception as e:  # the PR must still be opened
        print(f"(summary unavailable: {type(e).__name__})")
