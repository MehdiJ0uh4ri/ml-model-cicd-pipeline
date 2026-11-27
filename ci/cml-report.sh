#!/usr/bin/env bash
# Assemble one PR comment from whatever the run produced, and post it with CML.
#
# One comment, updated in place (`cml comment update`), not a new one per push:
# a PR with nine bot comments is a PR nobody reads. The comment is assembled
# from files on disk so it is identical whether CML posts it or a developer
# runs `cat artifacts/pr-comment.md` locally.
set -euo pipefail

OUT=artifacts/pr-comment.md
: >"$OUT"

{
  echo "# 🤖 Model pipeline — \`${GITHUB_SHA:-local}\`"
  echo
} >>"$OUT"

append() {
  local file=$1
  [[ -f $file ]] && { cat "$file" >>"$OUT"; echo >>"$OUT"; }
}

append artifacts/data_validation.md
append artifacts/report.md
append artifacts/shadow_report.md
append artifacts/canary_report.md
append artifacts/drift_report.md

if [[ -f artifacts/scorecard.json ]]; then
  {
    echo '<details><summary>Calibration (reliability curve)</summary>'
    echo
    echo '| bin | count | predicted | observed |'
    echo '|---|---:|---:|---:|'
    python - <<'PY'
import json
for row in json.load(open("artifacts/scorecard.json"))["calibration"]:
    print(f"| {row['bin']} | {row['count']} | {row['mean_predicted']:.3f} | {row['observed_rate']:.3f} |")
PY
    echo
    echo '</details>'
  } >>"$OUT"
fi

{
  echo
  echo "---"
  echo "Thresholds: [\`thresholds.yaml\`](../blob/${GITHUB_SHA:-main}/thresholds.yaml) · "
  echo "Baseline: [\`baselines/champion.json\`](../blob/${GITHUB_SHA:-main}/baselines/champion.json)"
} >>"$OUT"

if command -v cml >/dev/null 2>&1 && [[ -n "${GITHUB_TOKEN:-}" ]]; then
  cml comment update --target=pr "$OUT"
else
  echo "[ci] cml unavailable or no token; comment left at $OUT"
  cat "$OUT"
fi
