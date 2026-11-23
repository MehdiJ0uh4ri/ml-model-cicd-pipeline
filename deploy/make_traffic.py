"""Build the replay file used by shadow and canary.

Traffic is drawn from the *held-out* split, so a shadow run never re-scores
rows the challenger was fitted on. Optionally shifted, which is how the
drift monitor and the auto-rollback path get something real to catch:

    --shift none      the held-out distribution as-is
    --shift mild      a plausible macro move (incomes down, utilisation up)
    --shift severe    the same move, hard enough to break calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.model.features import split

SHIFTS = {
    "none": {},
    "mild": {"income": 0.92, "utilization": 1.08, "late_payments_2y": 1.15},
    "severe": {"income": 0.70, "utilization": 1.35, "late_payments_2y": 1.90},
}

PAYLOAD_FIELDS = [
    "application_id",
    "income",
    "debt_ratio",
    "credit_lines",
    "late_payments_2y",
    "utilization",
    "account_age_months",
    "employment",
    "region",
    "housing",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="write a JSONL traffic replay file")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/raw/applications.parquet"))
    ap.add_argument("--out", type=Path, default=Path("traffic/replay.jsonl"))
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--shift", choices=sorted(SHIFTS), default="none")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    _, test_df = split(pd.read_parquet(args.data), params["data"]["test_size"])

    rng = np.random.default_rng(args.seed)
    # Replay volume is driven by the canary's sample-size guard, which needs
    # more requests than the held-out split has rows. Repeats are honest here:
    # production replays the same customer more than once too, and a repeated
    # application_id deterministically lands on the same arm.
    n = args.n
    sample = test_df.iloc[rng.choice(len(test_df), size=n, replace=n > len(test_df))].copy()

    for column, factor in SHIFTS[args.shift].items():
        sample[column] = sample[column] * factor
    if args.shift != "none":
        sample["utilization"] = sample["utilization"].clip(0.0, 1.0)
        sample["late_payments_2y"] = sample["late_payments_2y"].round().astype(int)
        # Replayed traffic must not collide with the unshifted run in the
        # shadow log, or a comparison silently mixes two populations.
        sample["application_id"] = sample["application_id"] + f"-{args.shift}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in sample[PAYLOAD_FIELDS].to_dict(orient="records"):
            fh.write(json.dumps({k: _clean(v) for k, v in row.items()}) + "\n")

    # The labels travel beside the traffic, never inside it: serving must not
    # be able to see them, but the drift and canary reports need them.
    labels = {r["application_id"]: int(r["default"]) for r in sample.to_dict(orient="records")}
    label_path = args.out.with_suffix(".labels.json")
    label_path.write_text(json.dumps(labels, indent=0) + "\n", encoding="utf-8")

    print(f"[traffic] wrote {n} requests (shift={args.shift}) -> {args.out}")
    return 0


def _clean(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        # A missing income is `null` in the payload, never NaN: `NaN` is not
        # valid JSON, and every HTTP client rejects it. The service imputes
        # it exactly as training did.
        return None if np.isnan(value) else round(float(value), 6)
    return value if isinstance(value, (int, str)) else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
