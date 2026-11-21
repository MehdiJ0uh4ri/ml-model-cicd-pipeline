"""MLflow, with a local fallback that writes the same record shape.

Training must not fail because a tracking server is unreachable -- but a run
that logged nowhere must not be indistinguishable from one that logged to the
server either. So: MLflow when it imports and the tracking URI is reachable,
otherwise a JSON run file under `mlruns-local/`, and the backend is stamped
into every artifact so the gate and the PR comment can say which one produced
the numbers.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

LOCAL_DIR = Path(os.environ.get("MLFLOW_LOCAL_DIR", "mlruns-local"))


class Run:
    """Minimal surface used by train.py: params, metrics, artifacts, tags."""

    def __init__(self, backend: str, run_id: str, experiment: str):
        self.backend = backend
        self.run_id = run_id
        self.experiment = experiment
        self.params: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.tags: dict[str, str] = {}
        self.artifacts: list[str] = []
        self._mlflow = None

    # -- logging -----------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        flat = _flatten(params)
        self.params.update(flat)
        if self._mlflow:
            self._mlflow.log_params({k: str(v)[:500] for k, v in flat.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v is not None}
        self.metrics.update(clean)
        if self._mlflow:
            self._mlflow.log_metrics(clean, step=step)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update({k: str(v) for k, v in tags.items()})
        if self._mlflow:
            self._mlflow.set_tags(self.tags)

    def log_artifact(self, path: str | Path) -> None:
        self.artifacts.append(str(path))
        if self._mlflow:
            self._mlflow.log_artifact(str(path))

    def log_model(self, pipeline, example=None, artifact_path: str = "model") -> str | None:
        """Log the sklearn pipeline. Returns the model URI when MLflow is live.

        The input example is not decoration: it makes MLflow infer and store a
        signature, so a registered version carries its own schema and a
        payload with the wrong columns fails at load time rather than
        producing confident nonsense in serving.
        """
        if not self._mlflow:
            return None
        import mlflow.sklearn

        mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path, input_example=example)
        return f"runs:/{self.run_id}/{artifact_path}"

    # -- record ------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "backend": self.backend,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _mlflow_available() -> bool:
    if os.environ.get("MLFLOW_DISABLE") == "1":
        return False
    try:
        import mlflow  # noqa: F401
    except ImportError:
        return False
    return True


@contextmanager
def start_run(experiment: str, run_name: str, tags: dict[str, str] | None = None):
    """Yield a Run. Never raises because of tracking; degrades instead."""
    tags = tags or {}

    if _mlflow_available():
        import mlflow

        try:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
            mlflow.set_experiment(experiment)
            with mlflow.start_run(run_name=run_name) as active:
                run = Run("mlflow", active.info.run_id, experiment)
                run._mlflow = mlflow
                run.set_tags(tags)
                yield run
                _persist(run)
                return
        except Exception as exc:  # noqa: BLE001 - tracking is not the job
            print(
                f"[tracking] mlflow unusable ({type(exc).__name__}: {exc}); "
                "falling back to the local run store"
            )

    run = Run("local", uuid.uuid4().hex, experiment)
    run.set_tags(tags)
    yield run
    _persist(run)


def _persist(run: Run) -> None:
    """Always write a local copy of the run record.

    CI reads this file to build the CML comment; going through the MLflow API
    for it would make the PR comment depend on a reachable tracking server.
    """
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    record = run.to_dict() | {"logged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (LOCAL_DIR / f"{run.run_id}.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (LOCAL_DIR / "latest.json").write_text(
        json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8"
    )
