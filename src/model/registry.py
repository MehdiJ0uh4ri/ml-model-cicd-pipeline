"""Model registry: register a passing model, move aliases, roll back.

Alias model (MLflow >= 2.9 deprecated stages):

    challenger  the version shadow and canary exercise
    champion    the version serving loads for real traffic
    rollback    the previous champion, kept addressable so a rollback is one
                alias move rather than an archaeology exercise in the UI

The registry is the single place that answers "what is in production right
now". `python -m src.model.registry status` is the same question serving asks
at startup, which is why the local JSON backend mirrors the alias semantics
exactly instead of inventing a simpler scheme.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import joblib
import yaml

LOCAL_REGISTRY = Path("registry")


# ---------------------------------------------------------------- backends


class LocalRegistry:
    """File-backed registry. Same operations, no server.

    Versions are immutable directories; aliases are a single JSON file of
    pointers. That is the property that matters -- promotion and rollback are
    pointer moves, never a rebuild of the artifact.
    """

    def __init__(self, root: Path, model_name: str):
        self.root = root / model_name
        self.model_name = model_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write({"model": model_name, "versions": [], "aliases": {}, "history": []})

    def _read(self) -> dict:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.index_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def register(self, model_path: Path, meta: dict) -> str:
        idx = self._read()
        version = str(len(idx["versions"]) + 1)
        vdir = self.root / f"v{version}"
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, vdir / "model.joblib")
        (vdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        idx["versions"].append({"version": version, "path": str(vdir), **meta})
        idx["history"].append({"at": _now(), "action": "register", "version": version})
        self._write(idx)
        return version

    def set_alias(self, alias: str, version: str) -> None:
        idx = self._read()
        if not any(v["version"] == version for v in idx["versions"]):
            raise SystemExit(f"[registry] no version {version} of {self.model_name}")
        previous = idx["aliases"].get(alias)
        idx["aliases"][alias] = version
        idx["history"].append(
            {"at": _now(), "action": "set_alias", "alias": alias, "from": previous, "to": version}
        )
        self._write(idx)

    def alias_version(self, alias: str) -> str | None:
        return self._read()["aliases"].get(alias)

    def load_version(self, version: str) -> tuple[Any, dict]:
        """Return (pipeline, meta) for a version. The shape serving needs."""
        artifact = joblib.load(self.root / f"v{version}" / "model.joblib")
        return artifact["pipeline"], artifact["meta"]

    def status(self) -> dict:
        idx = self._read()
        return {
            "backend": "local",
            "model": self.model_name,
            "versions": len(idx["versions"]),
            "aliases": idx["aliases"],
            "history": idx["history"][-10:],
        }


class MlflowRegistry:
    def __init__(self, model_name: str):
        import mlflow
        from mlflow import MlflowClient

        self.mlflow = mlflow
        self.client = MlflowClient()
        self.model_name = model_name

    def register(self, model_path: Path, meta: dict) -> str:
        # model_path is part of the shared interface but unused here: MLflow
        # registers from the run's logged artifact, not from a local file, so
        # that the registered version and the tracked run cannot diverge.
        del model_path
        uri = meta.get("model_uri")
        if not uri:
            raise SystemExit(
                "[registry] mlflow backend needs a model_uri from the training run; "
                "train with MLflow enabled or use --backend local"
            )
        mv = self.mlflow.register_model(model_uri=uri, name=self.model_name)
        for key, value in meta.items():
            self.client.set_model_version_tag(self.model_name, mv.version, key, str(value)[:500])
        return str(mv.version)

    def set_alias(self, alias: str, version: str) -> None:
        self.client.set_registered_model_alias(self.model_name, alias, version)

    def alias_version(self, alias: str) -> str | None:
        try:
            return str(self.client.get_model_version_by_alias(self.model_name, alias).version)
        except Exception:  # noqa: BLE001 - a missing alias is a normal state
            return None

    def load_version(self, version: str) -> tuple[Any, dict]:
        """Pull the model out of the tracking store and rebuild serving's meta.

        The local backend keeps `meta` beside the artifact; MLflow keeps it as
        version tags, written by `register`. Both come back in the same shape
        so serving/app.py has no idea which registry it is talking to.
        """
        import mlflow.sklearn

        pipeline = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")
        tags = dict(self.client.get_model_version(self.model_name, version).tags)
        meta = {
            "run_id": tags.get("run_id", "unknown"),
            "git_sha": tags.get("git_sha", "unknown"),
            "snapshot": tags.get("snapshot", "unknown"),
            # Tags are strings. A threshold that silently became "0.38" rather
            # than 0.38 would compare as a string and never fire.
            "decision_threshold": float(tags["decision_threshold"])
            if "decision_threshold" in tags
            else None,
        }
        return pipeline, meta

    def status(self) -> dict:
        aliases = {}
        for alias in ("champion", "challenger", "rollback"):
            v = self.alias_version(alias)
            if v:
                aliases[alias] = v
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        return {
            "backend": "mlflow",
            "model": self.model_name,
            "versions": len(versions),
            "aliases": aliases,
        }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_registry(model_name: str, backend: str):
    if backend == "auto":
        try:
            import mlflow  # noqa: F401

            backend = "mlflow"
        except ImportError:
            backend = "local"
    if backend == "mlflow":
        try:
            return MlflowRegistry(model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[registry] mlflow unavailable ({exc}); using local registry")
    return LocalRegistry(LOCAL_REGISTRY, model_name)


# ---------------------------------------------------------------- commands


def cmd_register(args, params) -> int:
    """Register only what the gate approved. The verdict file is the authority."""
    verdict = json.loads(args.verdict.read_text(encoding="utf-8"))
    if verdict["decision"] != "PROMOTE" and not args.force:
        raise SystemExit(
            f"[registry] refusing to register: gate decision is {verdict['decision']} "
            f"({verdict['checks_failed']} failing checks)"
        )

    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    train_metrics = (
        json.loads(args.train_metrics.read_text(encoding="utf-8"))
        if args.train_metrics.exists()
        else {}
    )
    meta = {
        "git_sha": verdict["git_sha"],
        "run_id": verdict["run_id"],
        "snapshot": verdict["snapshot"],
        "roc_auc": scorecard["metrics"]["roc_auc"],
        "pr_auc": scorecard["metrics"]["pr_auc"],
        "recall": scorecard["metrics"]["recall"],
        "brier_score": scorecard["metrics"]["brier_score"],
        "decision_threshold": scorecard["model"]["decision_threshold"],
        "model_uri": train_metrics.get("model_uri"),
        "registered_at": _now(),
    }

    reg = get_registry(params["registry"]["model_name"], args.backend)
    version = reg.register(args.model, meta)
    # New versions land as challenger. Nothing becomes champion without
    # passing shadow and canary first -- that is deploy/promote.py's call.
    reg.set_alias(params["registry"]["challenger_alias"], version)

    out = {"version": version, "alias": params["registry"]["challenger_alias"], **meta}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[registry] registered {params['registry']['model_name']} v{version} as challenger")
    return 0


def cmd_promote(args, params) -> int:
    """Move champion to the given version, keeping the old one as rollback."""
    reg = get_registry(params["registry"]["model_name"], args.backend)
    champion_alias = params["registry"]["champion_alias"]
    version = args.version or reg.alias_version(params["registry"]["challenger_alias"])
    if not version:
        raise SystemExit("[registry] nothing to promote: no challenger alias set")

    previous = reg.alias_version(champion_alias)
    if previous == version:
        print(f"[registry] {champion_alias} already points at v{version}")
        return 0
    if previous:
        reg.set_alias("rollback", previous)
    reg.set_alias(champion_alias, version)
    if previous:
        print(f"[registry] champion v{previous} -> v{version} (rollback -> v{previous})")
    else:
        # First model: there is nothing to roll back to, and pretending
        # otherwise would leave a rollback alias pointing at the live model.
        print(f"[registry] champion set to v{version} (first model; no rollback target)")
    return 0


def cmd_rollback(args, params) -> int:
    reg = get_registry(params["registry"]["model_name"], args.backend)
    champion_alias = params["registry"]["champion_alias"]
    target = reg.alias_version("rollback")
    if not target:
        raise SystemExit("[registry] no rollback alias; nothing to roll back to")
    current = reg.alias_version(champion_alias)
    reg.set_alias(champion_alias, target)
    # Swap, do not clear: rolling back twice must not walk off a cliff.
    if current:
        reg.set_alias("rollback", current)
    print(f"[registry] rolled back champion v{current} -> v{target}")
    return 0


def cmd_status(args, params) -> int:
    reg = get_registry(params["registry"]["model_name"], args.backend)
    print(json.dumps(reg.status(), indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="model registry operations")
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--backend", choices=["auto", "mlflow", "local"], default="auto")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("register", help="register a gate-approved model as challenger")
    p.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    p.add_argument("--verdict", type=Path, default=Path("artifacts/verdict.json"))
    p.add_argument("--scorecard", type=Path, default=Path("artifacts/scorecard.json"))
    p.add_argument("--train-metrics", type=Path, default=Path("artifacts/train_metrics.json"))
    p.add_argument("--json-out", type=Path, default=Path("artifacts/registered.json"))
    p.add_argument("--force", action="store_true", help="register despite a BLOCKED verdict")
    p.set_defaults(fn=cmd_register)

    p = sub.add_parser("promote", help="point champion at a version (default: challenger)")
    p.add_argument("--version", default=None)
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("rollback", help="swap champion back to the previous version")
    p.set_defaults(fn=cmd_rollback)

    p = sub.add_parser("status", help="what is champion / challenger right now")
    p.set_defaults(fn=cmd_status)

    args = ap.parse_args()
    params = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    return args.fn(args, params)


if __name__ == "__main__":
    raise SystemExit(main())
