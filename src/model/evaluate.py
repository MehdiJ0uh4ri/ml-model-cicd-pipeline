"""Score the trained model on the held-out split and write a scorecard.

evaluate.py measures. It never decides -- that is gate.py's job. Keeping the
two apart means a threshold change is a diff in thresholds.yaml and gate.py,
and never a quiet edit to how a metric is computed.

The scorecard carries everything the gate needs to justify its verdict:
aggregate metrics, per-group metrics on the protected attribute, calibration,
and the integrity facts (row counts, data hash) that separate "the model got
worse" from "the measurement broke".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.model.features import TARGET, feature_columns, split


def core_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    out = {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, proba)),
        "positive_rate": float(pred.mean()),
        "base_rate": float(y_true.mean()),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out |= {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()}


def calibration_bins(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict]:
    """Reliability curve. Serving compares a probability to a fixed threshold,
    so a model whose 0.42 does not mean 42% is broken regardless of its AUC."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "count": int(sel.sum()),
                "mean_predicted": round(float(proba[sel].mean()), 6),
                "observed_rate": round(float(y_true[sel].mean()), 6),
            }
        )
    return rows


def group_metrics(
    df: pd.DataFrame, proba: np.ndarray, threshold: float, group_col: str
) -> dict[str, dict]:
    """Per-slice metrics on the protected attribute.

    Aggregate AUC can look healthy while one age band is scored at chance.
    Reporting each group separately is what lets thresholds.yaml put a floor
    under the worst one instead of the average.
    """
    y = df[TARGET].to_numpy()
    pred = (proba >= threshold).astype(int)
    out: dict[str, dict] = {}
    for value in sorted(map(str, pd.unique(df[group_col]))):
        sel = (df[group_col].astype(str) == value).to_numpy()
        n = int(sel.sum())
        yg = y[sel]
        entry = {
            "support": n,
            "selection_rate": round(float(pred[sel].mean()), 6) if n else None,
            "base_rate": round(float(yg.mean()), 6) if n else None,
        }
        # AUC is undefined for a single-class slice; report None rather than a
        # fabricated 0.5, and let the gate treat it as insufficient support.
        if n and 0 < yg.sum() < n:
            entry["roc_auc"] = round(float(roc_auc_score(yg, proba[sel])), 6)
            entry["recall"] = round(float(recall_score(yg, pred[sel], zero_division=0)), 6)
            entry["precision"] = round(float(precision_score(yg, pred[sel], zero_division=0)), 6)
        else:
            entry["roc_auc"] = None
            entry["recall"] = None
            entry["precision"] = None
        out[value] = entry
    return out


def fairness_summary(groups: dict[str, dict], min_support: int) -> dict:
    eligible = {k: v for k, v in groups.items() if v["support"] >= min_support}
    sel = [v["selection_rate"] for v in eligible.values() if v["selection_rate"] is not None]
    tpr = [v["recall"] for v in eligible.values() if v["recall"] is not None]
    aucs = {k: v["roc_auc"] for k, v in eligible.items() if v["roc_auc"] is not None}
    worst = min(aucs, key=aucs.get) if aucs else None
    return {
        "groups_evaluated": sorted(eligible),
        "groups_skipped_low_support": sorted(set(groups) - set(eligible)),
        "demographic_parity_diff": round(max(sel) - min(sel), 6) if len(sel) > 1 else 0.0,
        "equal_opportunity_diff": round(max(tpr) - min(tpr), 6) if len(tpr) > 1 else 0.0,
        "worst_group": worst,
        "worst_group_roc_auc": aucs[worst] if worst else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="score the model and write a scorecard")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--thresholds", type=Path, default=Path("thresholds.yaml"))
    ap.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    ap.add_argument("--data", type=Path, default=Path("data/raw/applications.parquet"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/scorecard.json"))
    args = ap.parse_args()

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    artifact = joblib.load(args.model)
    pipeline, meta = artifact["pipeline"], artifact["meta"]

    df = pd.read_parquet(args.data)
    _, test_df = split(df, params["data"]["test_size"])
    cols = feature_columns(params)
    threshold = float(meta["decision_threshold"])

    proba = pipeline.predict_proba(test_df[cols])[:, 1]
    y = test_df[TARGET].to_numpy()

    group_col = thresholds["fairness"]["group_column"]
    groups = group_metrics(test_df, proba, threshold, group_col)

    meta_path = args.data.with_suffix(".meta.json")
    current_hash = (
        json.loads(meta_path.read_text(encoding="utf-8")).get("content_hash")
        if meta_path.exists()
        else None
    )

    scorecard = {
        "model": {
            "run_id": meta["run_id"],
            "git_sha": meta["git_sha"],
            "snapshot": meta["snapshot"],
            "decision_threshold": threshold,
            "sklearn_version": meta["sklearn_version"],
            "trained_at": meta["trained_at"],
        },
        "integrity": {
            "test_rows": int(len(test_df)),
            "prediction_nan_rate": round(float(np.isnan(proba).mean()), 8),
            "train_data_hash": meta["data_content_hash"],
            "eval_data_hash": current_hash,
            "data_hash_matches": current_hash == meta["data_content_hash"],
        },
        "metrics": core_metrics(y, proba, threshold),
        "groups": {"column": group_col, "values": groups}
        | fairness_summary(groups, thresholds["fairness"]["min_group_support"]),
        "calibration": calibration_bins(y, proba),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")

    m = scorecard["metrics"]
    print(
        f"[eval] rows={len(test_df)} roc_auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} "
        f"recall={m['recall']:.4f} brier={m['brier_score']:.4f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
