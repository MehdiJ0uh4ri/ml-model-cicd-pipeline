"""Write the committed baseline that the regression gate compares against.

Called by CI on `main` only, after a model has actually become champion. The
baseline is the champion's scorecard -- writing it any earlier would let a
model that failed canary set the bar for the next one.

`baselines/champion.json` is a tracked file, CODEOWNERS-gated like
thresholds.yaml. Regenerating it by hand to make a red build green is the one
move this repo is designed to make visible in review.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="promote a scorecard to the tracked baseline")
    ap.add_argument("--scorecard", type=Path, default=Path("artifacts/scorecard.json"))
    ap.add_argument("--verdict", type=Path, default=Path("artifacts/verdict.json"))
    ap.add_argument("--baseline", type=Path, default=Path("baselines/champion.json"))
    ap.add_argument("--registered", type=Path, default=Path("artifacts/registered.json"))
    ap.add_argument("--force", action="store_true", help="write even on a BLOCKED verdict")
    args = ap.parse_args()

    verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
    if verdict["decision"] != "PROMOTE" and not args.force:
        raise SystemExit(
            f"[baseline] refusing to move the bar: gate decision is {verdict['decision']}"
        )

    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    registered = (
        json.loads(args.registered.read_text(encoding="utf-8")) if args.registered.exists() else {}
    )

    previous = None
    if args.baseline.exists():
        previous = json.loads(args.baseline.read_text(encoding="utf-8"))

    baseline = {
        "_comment": (
            "Champion scorecard. Written by ci/promote-baseline.sh on main after promotion. "
            "Do not edit by hand -- see docs/thresholds.md."
        ),
        "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "registry_version": registered.get("version"),
        "model": scorecard["model"],
        "metrics": scorecard["metrics"],
        "groups": {
            "column": scorecard["groups"]["column"],
            "demographic_parity_diff": scorecard["groups"]["demographic_parity_diff"],
            "equal_opportunity_diff": scorecard["groups"]["equal_opportunity_diff"],
            "worst_group": scorecard["groups"]["worst_group"],
            "worst_group_roc_auc": scorecard["groups"]["worst_group_roc_auc"],
        },
        "supersedes": (
            {
                "git_sha": previous["model"]["git_sha"],
                "roc_auc": previous["metrics"]["roc_auc"],
                "promoted_at": previous.get("promoted_at"),
            }
            if previous
            else None
        ),
    }

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    before = previous["metrics"]["roc_auc"] if previous else None
    print(
        f"[baseline] {args.baseline}: roc_auc "
        f"{before if before is not None else 'none'} -> {baseline['metrics']['roc_auc']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
