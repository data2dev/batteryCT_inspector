"""Command-line interface: ``batteryct``."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from batteryct import __version__
from batteryct.io import save_image
from batteryct.synthetic import make_cohort


def _cmd_ingest(args: argparse.Namespace) -> int:
    import yaml

    from batteryct.ingest import IngestConfig, run_ingest, write_kaggle_metadata

    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}
    if args.limit_scans is not None:
        raw["limit_scans"] = args.limit_scans
    config = IngestConfig.from_mapping(raw)
    manifest = run_ingest(config, args.out, workers=args.workers)
    if args.kaggle_id:
        write_kaggle_metadata(args.out, args.kaggle_id, f"batteryct {config.cohort} views")
    print(f"manifest: {manifest}")
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = make_cohort(args.n_normal, args.n_per_defect, size=args.size, seed=args.seed)
    with (out / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cell_id", "file", "defect", "seed", "synthetic"])
        for i, cell in enumerate(cells):
            cell_id = f"synth_{i:04d}"
            name = f"{cell_id}.png"
            save_image(cell.image, out / name)
            writer.writerow([cell_id, name, cell.defect, cell.seed, "true"])
    print(f"wrote {len(cells)} SYNTHETIC images to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batteryct", description="Anomaly screening for battery CT cross-sections."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="write a SYNTHETIC test cohort (not real data)")
    synth.add_argument("out", help="output directory")
    synth.add_argument("--n-normal", type=int, default=20)
    synth.add_argument("--n-per-defect", type=int, default=2)
    synth.add_argument("--size", type=int, default=128)
    synth.add_argument("--seed", type=int, default=0)
    synth.set_defaults(func=_cmd_synth)

    ingest = sub.add_parser(
        "ingest", help="fetch selected views of one cohort from Figshare (range requests)"
    )
    ingest.add_argument("config", help="ingest YAML config, e.g. configs/ingest/na18650.yaml")
    ingest.add_argument("out", help="output directory")
    ingest.add_argument("--limit-scans", type=int, default=None, help="only the first N scans")
    ingest.add_argument("--workers", type=int, default=8)
    ingest.add_argument(
        "--kaggle-id", default=None, help="also write dataset-metadata.json for <user>/<slug>"
    )
    ingest.set_defaults(func=_cmd_ingest)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
