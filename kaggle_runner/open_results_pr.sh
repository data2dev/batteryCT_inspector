#!/usr/bin/env bash
# Open a PR adding results/runs/<run_id>/ unless one already exists.
# Usage: kaggle_runner/open_results_pr.sh RUN_ID STATE
set -euo pipefail
run_id="$1"; state="${2:-complete}"
dir="results/runs/$run_id"
branch="results/$run_id"

[ -d "$dir" ] || { echo "no $dir, nothing to do"; exit 0; }
if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
  echo "branch $branch already exists; skipping"; exit 0
fi
if git cat-file -e "origin/main:$dir/run_meta.json" 2>/dev/null; then
  echo "$dir already on main; skipping"; exit 0
fi

git config user.name "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
git checkout -b "$branch"
git add "$dir"
git commit -m "results: $run_id ($state)"
git push origin "$branch"

summary=$(python - "$dir" <<'EOF'
import csv, json, sys
from pathlib import Path
d = Path(sys.argv[1])
out = []
meta = json.loads((d / "run_meta.json").read_text())
out.append(f"- **status**: {meta.get('status')}  \n- **commit**: `{meta.get('git_sha', '')[:7]}`  \n- **runtime**: {meta.get('runtime_s')} s")
if (d / "metrics.json").exists():
    m = json.loads((d / "metrics.json").read_text())
    out.append(f"- **scored**: {m.get('n_scans_scored')} cells (dropped {m.get('n_scans_dropped')})")
    out.append(f"- **score quantiles**: {m.get('score_quantiles')}, max {m.get('score_max')}")
if (d / "scores.csv").exists():
    rows = list(csv.DictReader((d / "scores.csv").open()))[:10]
    out.append("\n| rank | scan | score | driving view |\n|---|---|---|---|")
    out += [f"| {r['rank']} | {r['scan_id']} | {r['score']} | {r['driving_view']} |" for r in rows]
print("\n".join(out))
EOF
)
body="Results collected from Kaggle run \`$run_id\` ($state).

$summary

Scores mean \"differs from the cohort\", not \"defective\", until reviewed.
Derived from the Glimpse battery CT dataset (CC BY-NC-SA 4.0, see results/LICENSE)."
gh pr create --base main --head "$branch" --title "results: $run_id" --body "$body" \
  || echo "::warning::Could not open PR (enable 'Allow GitHub Actions to create pull requests'). Branch $branch was pushed."
