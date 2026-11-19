"""Build the raw snapshot that DVC versions.

Real projects pull this from a warehouse. Here it is generated from a seed so
that `dvc repro` on two machines produces byte-identical parquet -- which is
the only way the "same data hash in training and evaluation" check in
src/model/gate.py can mean anything.

The generator can inject defects (nulls, label flips) on purpose. That is what
makes the Great Expectations gate demonstrable: `make demo-bad-data` raises
the null rate past the suite's tolerance and the pipeline stops before a
single epoch of training is spent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EMPLOYMENT = ["salaried", "self_employed", "contract", "retired", "unemployed"]
REGION = ["north", "south", "east", "west"]
HOUSING = ["own", "rent", "mortgage"]
AGE_GROUPS = ["18-25", "26-40", "41-60", "60+"]

SIGNAL_SCALE = 2.2
INTERCEPT = -2.8


def _params(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_frame(cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    n = int(cfg["n_rows"])

    age_group = rng.choice(AGE_GROUPS, size=n, p=[0.18, 0.37, 0.31, 0.14])
    employment = rng.choice(EMPLOYMENT, size=n, p=[0.52, 0.16, 0.14, 0.12, 0.06])

    # Income depends on age band and employment: the correlation is what makes
    # the fairness slice non-trivial. A model that leans on income inherits a
    # little of the age signal, and the per-group gate is what catches it.
    base = np.select(
        [age_group == "18-25", age_group == "26-40", age_group == "41-60"],
        [38_000, 46_000, 50_000],
        default=42_000,
    ).astype(float)
    emp_mult = np.select(
        [employment == "self_employed", employment == "contract", employment == "unemployed"],
        [1.10, 0.92, 0.45],
        default=1.0,
    )
    income = np.round(rng.lognormal(mean=0.0, sigma=0.32, size=n) * base * emp_mult, 2)

    utilization = np.clip(rng.beta(2.0, 3.2, size=n), 0.0, 1.0).round(4)
    credit_lines = rng.poisson(4.1, size=n).clip(0, 24)
    late_payments_2y = rng.poisson(0.55 + 1.9 * utilization).clip(0, 20)
    account_age_months = rng.integers(1, 300, size=n)
    debt_ratio = np.round(
        np.clip(rng.gamma(2.0, 0.18, size=n) + 0.45 * utilization, 0.0, 3.0), 4
    )

    # Log-odds of default. SIGNAL_SCALE and the intercept were fitted (see
    # docs/thresholds.md) so a well-specified model lands near ROC-AUC 0.85
    # with a ~27% positive rate: enough headroom above the floors that seed
    # noise does not block a good model, tight enough that a real regression
    # trips the gate instead of being absorbed by slack.
    z = (
        INTERCEPT
        + SIGNAL_SCALE
        * (
            1.65 * debt_ratio
            + 1.30 * utilization
            + 0.34 * late_payments_2y
            - 0.55 * np.log1p(income / 30_000.0)
            - 0.004 * account_age_months
            - 0.03 * credit_lines
        )
        + np.where(employment == "unemployed", 0.55, 0.0)
        # The only direct age effect. Small on purpose: the interesting
        # fairness signal is the indirect one through income, which no
        # per-feature audit of the model would surface.
        + np.where(age_group == "18-25", 0.10, 0.0)
    )
    p = 1.0 / (1.0 + np.exp(-z))
    default = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "application_id": [f"APP-{cfg['snapshot']}-{i:07d}" for i in range(n)],
            "income": income,
            "debt_ratio": debt_ratio,
            "credit_lines": credit_lines.astype("int64"),
            "late_payments_2y": late_payments_2y.astype("int64"),
            "utilization": utilization,
            "account_age_months": account_age_months.astype("int64"),
            "employment": pd.Categorical(employment, categories=EMPLOYMENT),
            "region": pd.Categorical(rng.choice(REGION, size=n), categories=REGION),
            "housing": pd.Categorical(
                rng.choice(HOUSING, size=n, p=[0.31, 0.40, 0.29]), categories=HOUSING
            ),
            "age_group": pd.Categorical(age_group, categories=AGE_GROUPS),
            "snapshot_date": cfg["snapshot"],
            "default": default.astype("int64"),
        }
    )

    inject = cfg.get("inject") or {}
    null_rate = float(inject.get("null_rate", 0.0))
    if null_rate > 0:
        # Nulls land in income only. A blanket sprinkle would trip every
        # expectation at once and teach nothing about which one fired.
        idx = rng.choice(n, size=int(n * null_rate), replace=False)
        df.loc[idx, "income"] = np.nan

    flip = float(inject.get("label_flip", 0.0))
    if flip > 0:
        idx = rng.choice(n, size=int(n * flip), replace=False)
        df.loc[idx, "default"] = 1 - df.loc[idx, "default"]

    return df


def frame_hash(df: pd.DataFrame) -> str:
    """Content hash of the snapshot, independent of parquet encoding."""
    h = hashlib.sha256()
    for col in sorted(df.columns):
        h.update(col.encode())
        h.update(pd.util.hash_pandas_object(df[col], index=False).values.tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/raw/applications.parquet"))
    ap.add_argument("--null-rate", type=float, default=None, help="override params.data.inject")
    ap.add_argument("--label-flip", type=float, default=None)
    args = ap.parse_args()

    cfg = _params(args.params)["data"]
    if args.null_rate is not None:
        cfg.setdefault("inject", {})["null_rate"] = args.null_rate
    if args.label_flip is not None:
        cfg.setdefault("inject", {})["label_flip"] = args.label_flip

    df = build_frame(cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    digest = frame_hash(df)
    meta = {
        "rows": len(df),
        "columns": list(df.columns),
        "snapshot": cfg["snapshot"],
        "seed": cfg["seed"],
        "inject": cfg.get("inject", {}),
        "content_hash": digest,
        "positive_rate": round(float(df["default"].mean()), 6),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}  rows={len(df)}  hash={digest}  pos_rate={meta['positive_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
