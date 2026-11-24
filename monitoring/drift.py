"""Drift monitoring over the shadow log: has production stopped looking like training?

Runs nightly, not on PRs. Drift is not caused by a commit, so blaming one is
pointless -- the nightly job opens an issue instead of failing somebody's
build.

Two families of signal, because they fail at different times:

  * **feature drift** (PSI, KS) -- the inputs moved. Visible immediately, with
    no labels required. This is the early warning.
  * **prediction drift** -- the score distribution moved. Catches shifts in
    feature *combinations* that per-feature tests miss entirely.

Evidently produces the same report with nicer HTML when it is installed; the
numeric verdict comes from the implementations here either way, so the
nightly job's decision does not depend on an optional dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.model.features import split

# PSI convention used across the industry: < 0.1 stable, 0.1-0.25 moderate,
# > 0.25 significant. The alert fires on the third band.
PSI_WARN = 0.10
PSI_ALERT = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index over quantile bins of the reference."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 2 or len(current) < 2:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_frac = np.histogram(current, bins=edges)[0] / len(current)
    # Floor the fractions: an empty bin makes the log term infinite, which
    # would turn one unseen value into an unbounded drift score.
    eps = 1e-6
    ref_frac = np.clip(ref_frac, eps, None)
    cur_frac = np.clip(cur_frac, eps, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance, without a scipy dependency."""
    reference = np.sort(reference[~np.isnan(reference)])
    current = np.sort(current[~np.isnan(current)])
    if len(reference) == 0 or len(current) == 0:
        return 0.0
    grid = np.union1d(reference, current)
    cdf_ref = np.searchsorted(reference, grid, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, grid, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def categorical_drift(reference: pd.Series, current: pd.Series) -> dict:
    ref = reference.astype(str).value_counts(normalize=True)
    cur = current.astype(str).value_counts(normalize=True)
    categories = ref.index.union(cur.index)
    ref_v = ref.reindex(categories, fill_value=0.0).to_numpy()
    cur_v = cur.reindex(categories, fill_value=0.0).to_numpy()
    eps = 1e-6
    score = float(
        np.sum((cur_v - ref_v) * np.log(np.clip(cur_v, eps, None) / np.clip(ref_v, eps, None)))
    )
    unseen = sorted(set(cur.index) - set(ref.index))
    return {"psi": round(score, 6), "unseen_categories": unseen}


def load_shadow_frame(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit(f"[drift] shadow log {path} is empty; nothing to compare")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="feature and prediction drift report")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--reference", type=Path, default=Path("data/raw/applications.parquet"))
    ap.add_argument("--current", type=Path, default=Path("traffic/replay.jsonl"))
    ap.add_argument("--shadow-log", type=Path, default=Path("artifacts/shadow_log.jsonl"))
    ap.add_argument("--json-out", type=Path, default=Path("artifacts/drift_report.json"))
    ap.add_argument("--markdown", type=Path, default=Path("artifacts/drift_report.md"))
    ap.add_argument("--fail-on-alert", action="store_true", help="exit non-zero when PSI alerts")
    args = ap.parse_args()

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    train_df, _ = split(pd.read_parquet(args.reference), params["data"]["test_size"])
    current = pd.DataFrame(
        [json.loads(line) for line in args.current.read_text(encoding="utf-8").splitlines() if line]
    )

    features = {}
    for col in params["features"]["numeric"]:
        if col not in current.columns:
            continue
        score = psi(train_df[col].to_numpy(dtype="float64"), current[col].to_numpy(dtype="float64"))
        features[col] = {
            "type": "numeric",
            "psi": round(score, 6),
            "ks": round(
                ks_statistic(
                    train_df[col].to_numpy(dtype="float64"),
                    current[col].to_numpy(dtype="float64"),
                ),
                6,
            ),
            "status": _band(score),
        }
    for col in params["features"]["categorical"]:
        if col not in current.columns:
            continue
        entry = categorical_drift(train_df[col], current[col])
        entry |= {"type": "categorical", "status": _band(entry["psi"])}
        if entry["unseen_categories"]:
            # An unseen category one-hot-encodes to all zeros: the model keeps
            # answering and the metric keeps looking fine. Always an alert.
            entry["status"] = "alert"
        features[col] = entry

    prediction = None
    if args.shadow_log.exists():
        shadow = load_shadow_frame(args.shadow_log)
        champ = shadow["champion_proba"].dropna().to_numpy(dtype="float64")
        chal = shadow["challenger_proba"].dropna().to_numpy(dtype="float64")
        prediction = {
            "champion_mean": round(float(champ.mean()), 6) if len(champ) else None,
            "challenger_mean": round(float(chal.mean()), 6) if len(chal) else None,
            "champion_vs_challenger_ks": round(ks_statistic(champ, chal), 6)
            if len(champ) and len(chal)
            else None,
            "rows": int(len(shadow)),
        }

    alerts = sorted(k for k, v in features.items() if v["status"] == "alert")
    warnings = sorted(k for k, v in features.items() if v["status"] == "warn")
    report = {
        "decision": "ALERT" if alerts else ("WARN" if warnings else "STABLE"),
        "reference": str(args.reference),
        "current": str(args.current),
        "reference_rows": int(len(train_df)),
        "current_rows": int(len(current)),
        "thresholds": {"psi_warn": PSI_WARN, "psi_alert": PSI_ALERT},
        "features": features,
        "prediction": prediction,
        "alerts": alerts,
        "warnings": warnings,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(_markdown(report), encoding="utf-8")

    print(
        f"[drift] {report['decision']}: {len(alerts)} alert(s), {len(warnings)} warning(s) "
        f"over {len(features)} features"
    )
    for name in alerts:
        print(f"  ALERT {name}: psi={features[name]['psi']}")

    return 1 if (alerts and args.fail_on_alert) else 0


def _band(score: float) -> str:
    if score >= PSI_ALERT:
        return "alert"
    if score >= PSI_WARN:
        return "warn"
    return "stable"


def _markdown(report: dict) -> str:
    icon = {"STABLE": "🟢", "WARN": "🟡", "ALERT": "🔴"}[report["decision"]]
    lines = [
        f"## {icon} Drift report — **{report['decision']}**",
        "",
        f"{report['current_rows']:,} recent requests vs "
        f"{report['reference_rows']:,} training rows. "
        f"PSI bands: warn ≥ {PSI_WARN}, alert ≥ {PSI_ALERT}.",
        "",
        "| feature | type | PSI | KS | status |",
        "|---|---|---:|---:|---|",
    ]
    for name, f in sorted(report["features"].items(), key=lambda kv: -kv[1]["psi"]):
        lines.append(
            f"| {name} | {f['type']} | {f['psi']:.4f} | "
            f"{f.get('ks', float('nan')):.4f} | {f['status']} |"
            if f["type"] == "numeric"
            else f"| {name} | {f['type']} | {f['psi']:.4f} | — | {f['status']} |"
        )
    p = report.get("prediction")
    if p:
        lines += [
            "",
            f"**Prediction drift** — champion mean {p['champion_mean']}, "
            f"challenger mean {p['challenger_mean']}, KS {p['champion_vs_challenger_ks']} "
            f"over {p['rows']:,} shadow rows.",
        ]
    if report["alerts"]:
        lines += [
            "",
            "Alerting features: " + ", ".join(f"`{a}`" for a in report["alerts"]),
            "",
            "Drift is not a build failure — no commit caused it. Triage per docs/runbook.md.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
