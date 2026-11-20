"""Check a Kaggle experiment kernel and copy finished results into results/runs/.

Usage:
    python kaggle_runner/collect.py KERNEL_ID [--dest results/runs] [--wait-minutes N]

Prints one line ``state=<queued|running|complete|error|missing> run_id=<id>`` and,
for a finished run, copies the small outputs to ``<dest>/<run_id>/``. Exit code is 0
for complete, 3 for still running/queued, 1 for error/missing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COLLECT = (
    "scores.csv",
    "metrics.json",
    "run_meta.json",
    "config.yaml",
    "top_outliers.png",
    "most_typical.png",
)
MAX_FILE_BYTES = 5_000_000  # results/ is for small files only
MAX_REVIEW_BYTES = 12_000_000  # whole review pack per run


def parse_status(text: str) -> str:
    t = text.lower()
    for state, words in (
        ("complete", ("complete",)),
        ("error", ("error", "cancel", "fail")),
        ("running", ("running",)),
        ("queued", ("queued", "new", "pending")),
    ):
        if any(re.search(rf"\b{w}", t) for w in words):
            return state
    return "missing"


def kaggle(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, check=False)


def status(kernel: str) -> str:
    p = kaggle("kernels", "status", kernel)
    return parse_status(p.stdout) if p.returncode == 0 else "missing"


def collect_outputs(src: Path, dest_root: Path) -> tuple[str, Path]:
    metas = sorted(src.rglob("run_meta.json"))
    if not metas:
        raise FileNotFoundError("kernel output has no run_meta.json")
    run_dir = metas[-1].parent
    run_id = json.loads(metas[-1].read_text())["run_id"]
    dest = dest_root / run_id
    dest.mkdir(parents=True, exist_ok=True)
    for name in COLLECT:
        f = run_dir / name
        if f.is_file() and f.stat().st_size <= MAX_FILE_BYTES:
            shutil.copy2(f, dest / name)
    review = run_dir / "review"
    if review.is_dir():  # review pack: index.md + one JPEG page per top cell
        total = 0
        (dest / "review").mkdir(exist_ok=True)
        for f in sorted(review.iterdir()):
            if f.suffix in (".md", ".jpg") and total + f.stat().st_size <= MAX_REVIEW_BYTES:
                shutil.copy2(f, dest / "review" / f.name)
                total += f.stat().st_size
    logs = sorted(src.glob("*.log"))
    if logs:  # keep the tail of the Kaggle log for debugging
        tail = logs[0].read_text(errors="replace")[-20_000:]
        (dest / "kaggle_log_tail.txt").write_text(tail)
    return run_id, dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kernel", help="<username>/<kernel-slug>")
    ap.add_argument("--dest", type=Path, default=Path("results/runs"))
    ap.add_argument("--wait-minutes", type=float, default=0.0)
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    a = ap.parse_args(argv)

    deadline = time.time() + 60 * a.wait_minutes
    state = status(a.kernel)
    while state in ("queued", "running") and time.time() < deadline:
        time.sleep(a.poll_seconds)
        state = status(a.kernel)

    run_id = ""
    if state in ("complete", "error"):
        with tempfile.TemporaryDirectory() as tmp:
            p = kaggle("kernels", "output", a.kernel, "-p", tmp)
            if p.returncode != 0:
                print(p.stderr, file=sys.stderr)
            try:
                run_id, dest = collect_outputs(Path(tmp), a.dest)
                print(f"collected into {dest}", file=sys.stderr)
            except FileNotFoundError as e:
                print(f"warning: {e}", file=sys.stderr)
    print(f"state={state} run_id={run_id}")
    return {"complete": 0, "queued": 3, "running": 3}.get(state, 1)


if __name__ == "__main__":
    sys.exit(main())
