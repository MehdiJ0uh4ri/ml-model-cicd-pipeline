"""Train one model, log one MLflow run, write one artifact.

Preconditions this script assumes and re-checks rather than trusts:

* the snapshot passed the expectation suite (artifacts/data_validation.json
  says PASS). Training on data that failed the gate is the exact failure mode
  the gate exists to prevent, so it is checked here too -- a developer running
  `python -m src.model.train` directly gets the same protection CI does;
* the data hash is recorded into the model artifact, so evaluation later can
  prove it scored the model on the snapshot it was trained against.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from sklearn.metrics import roc_auc_score

from src.model import tracking
from src.model.features import TARGET, as_frame, build_pipeline, feature_columns, split


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def data_hash(data_path: Path) -> str:
    meta = data_path.with_suffix(".meta.json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("content_hash", "unknown")
    return "unknown"


def require_validated_data(report_path: Path, strict: bool) -> None:
    if not report_path.exists():
        msg = f"no data validation report at {report_path}; run `make validate` first"
        if strict:
            raise SystemExit(f"[train] refusing to train: {msg}")
        print(f"[train] warning: {msg}")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("decision") != "PASS":
        raise SystemExit(
            f"[train] refusing to train: data gate reported {report.get('decision')} "
            f"({report.get('passed')}/{report.get('total')} expectations passed)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="train the credit-default model")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--data", type=Path, default=Path("data/raw/applications.parquet"))
    ap.add_argument(
        "--validation-report", type=Path, default=Path("artifacts/data_validation.json")
    )
    ap.add_argument("--out", type=Path, default=Path("artifacts/model.joblib"))
    ap.add_argument("--metrics-out", type=Path, default=Path("artifacts/train_metrics.json"))
    ap.add_argument("--experiment", default="credit-default-risk")
    ap.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="skip the data-gate precondition (local experimentation only; CI never sets it)",
    )
    args = ap.parse_args()

    require_validated_data(args.validation_report, strict=not args.allow_unvalidated)

    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    df = pd.read_parquet(args.data)
    train_df, test_df = split(df, params["data"]["test_size"])
    cols = feature_columns(params)

    run_name = f"{params['data']['snapshot']}-{git_sha()}"
    with tracking.start_run(args.experiment, run_name, tags={"git_sha": git_sha()}) as run:
        run.log_params(
            {
                "data": {
                    "snapshot": params["data"]["snapshot"],
                    "rows_total": len(df),
                    "rows_train": len(train_df),
                    "rows_test": len(test_df),
                    "content_hash": data_hash(args.data),
                    "test_size": params["data"]["test_size"],
                },
                "model": params["model"],
                "features": {"columns": ",".join(cols)},
                "env": {
                    "sklearn": sklearn.__version__,
                    "python": platform.python_version(),
                },
            }
        )

        pipeline = build_pipeline(params)
        started = time.perf_counter()
        pipeline.fit(train_df[cols], train_df[TARGET])
        fit_seconds = time.perf_counter() - started

        # Training-set metrics only. The numbers that decide promotion come
        # from evaluate.py on the held-out split; logging train scores here is
        # for spotting overfit in the MLflow UI, not for gating.
        train_auc = float(
            roc_auc_score(train_df[TARGET], pipeline.predict_proba(train_df[cols])[:, 1])
        )
        n_iter = getattr(pipeline.named_steps["clf"], "n_iter_", None)
        run.log_metrics(
            {
                "fit_seconds": fit_seconds,
                "train_roc_auc": train_auc,
                "n_iter": float(n_iter) if n_iter is not None else None,
                "rows_train": float(len(train_df)),
            }
        )

        artifact = {
            "pipeline": pipeline,
            "meta": {
                "run_id": run.run_id,
                "backend": run.backend,
                "experiment": args.experiment,
                "git_sha": git_sha(),
                "snapshot": params["data"]["snapshot"],
                "data_content_hash": data_hash(args.data),
                "feature_columns": cols,
                # Serving must threshold at the same value evaluation used.
                # Carrying it in the artifact removes the chance of a config
                # file in the serving repo disagreeing with the scorecard.
                "decision_threshold": params["model"]["decision_threshold"],
                "sklearn_version": sklearn.__version__,
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.out)
        run.log_artifact(args.out)
        # The signature is inferred from a *serving-shaped* example: the same
        # as_frame() path an HTTP payload takes, not the parquet frame with
        # its categorical dtypes. A signature that does not match what
        # production sends is worse than none.
        example = as_frame(train_df[cols].head(5).to_dict(orient="records"), params)
        model_uri = run.log_model(pipeline, example=example)

        metrics = {
            "run_id": run.run_id,
            "backend": run.backend,
            "model_uri": model_uri,
            "fit_seconds": round(fit_seconds, 3),
            "train_roc_auc": round(train_auc, 6),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "data_content_hash": data_hash(args.data),
        }
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(
        f"[train] {run.backend} run={run.run_id} fit={fit_seconds:.1f}s "
        f"train_auc={train_auc:.4f} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
