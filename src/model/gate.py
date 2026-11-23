"""The evaluation gate: turn a scorecard into a promote / block decision.

Rules, in the order they are evaluated:

1. **integrity** -- did the measurement work at all? A short test set or a
   data-hash mismatch fails here with its own message, so a broken pipeline
   never gets reported as a quality regression;
2. **floors** -- absolute minimums from thresholds.yaml. Applied even when
   there is no baseline (first model ever, or `--no-baseline`);
3. **regression** -- comparison against the committed champion scorecard,
   with per-metric tolerances;
4. **fairness** -- per-group floors plus parity spreads.

Every check produces a row in the report whether it passed or failed. A gate
that only prints failures teaches nobody what it is actually protecting.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Check:
    name: str
    category: str
    passed: bool
    observed: float | int | str | None
    limit: float | int | str | None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "passed": self.passed,
            "observed": self.observed,
            "limit": self.limit,
            "detail": self.detail,
        }


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    @property
    def decision(self) -> str:
        return "BLOCKED" if self.failures else "PROMOTE"


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def check_integrity(sc: dict, cfg: dict, report: Report) -> None:
    integ = sc["integrity"]
    report.add(
        Check(
            "test_rows",
            "integrity",
            integ["test_rows"] >= cfg["min_test_rows"],
            integ["test_rows"],
            cfg["min_test_rows"],
            "too few held-out rows for the metrics to mean anything",
        )
    )
    report.add(
        Check(
            "prediction_nan_rate",
            "integrity",
            integ["prediction_nan_rate"] <= cfg["max_prediction_nan_rate"],
            integ["prediction_nan_rate"],
            cfg["max_prediction_nan_rate"],
            "the model emitted NaN probabilities",
        )
    )
    if cfg.get("require_matching_data_hash", True):
        report.add(
            Check(
                "data_hash_matches",
                "integrity",
                bool(integ["data_hash_matches"]),
                integ["eval_data_hash"],
                integ["train_data_hash"],
                "scored on a different snapshot than the one trained on",
            )
        )


def check_floors(sc: dict, cfg: dict, report: Report) -> None:
    m = sc["metrics"]
    for metric in ("roc_auc", "pr_auc", "accuracy", "recall"):
        if metric not in cfg:
            continue
        report.add(
            Check(
                f"floor.{metric}",
                "floor",
                m[metric] >= cfg[metric],
                m[metric],
                cfg[metric],
                "below the absolute minimum for release",
            )
        )
    if "brier_score_max" in cfg:
        report.add(
            Check(
                "floor.brier_score",
                "floor",
                m["brier_score"] <= cfg["brier_score_max"],
                m["brier_score"],
                cfg["brier_score_max"],
                "probabilities are too poorly calibrated to threshold in serving",
            )
        )


def check_regression(sc: dict, baseline: dict | None, cfg: dict, report: Report) -> None:
    if baseline is None:
        report.add(
            Check(
                "baseline.present",
                "regression",
                True,
                "none",
                "none",
                "no committed baseline; floors only (first release or --no-baseline)",
            )
        )
        return

    cur, base = sc["metrics"], baseline["metrics"]
    drops = {
        "roc_auc": cfg.get("roc_auc_drop_max"),
        "pr_auc": cfg.get("pr_auc_drop_max"),
        "recall": cfg.get("recall_drop_max"),
    }
    for metric, allowed in drops.items():
        if allowed is None:
            continue
        delta = cur[metric] - base[metric]
        report.add(
            Check(
                f"regression.{metric}",
                "regression",
                delta >= -allowed,
                round(delta, 6),
                -allowed,
                f"champion {base[metric]:.4f} -> challenger {cur[metric]:.4f}",
            )
        )
    if cfg.get("brier_rise_max") is not None:
        delta = cur["brier_score"] - base["brier_score"]
        report.add(
            Check(
                "regression.brier_score",
                "regression",
                delta <= cfg["brier_rise_max"],
                round(delta, 6),
                cfg["brier_rise_max"],
                f"champion {base['brier_score']:.4f} -> challenger {cur['brier_score']:.4f}",
            )
        )


def check_fairness(sc: dict, cfg: dict, report: Report) -> None:
    groups = sc["groups"]
    evaluated = groups["groups_evaluated"]

    # No eligible slice means the fairness gate measured nothing. That is a
    # blocked release, not a silent pass.
    report.add(
        Check(
            "fairness.coverage",
            "fairness",
            len(evaluated) >= 2,
            len(evaluated),
            2,
            f"slices with >= {cfg['min_group_support']} rows: {evaluated or 'none'}",
        )
    )

    for name in evaluated:
        auc = groups["values"][name]["roc_auc"]
        report.add(
            Check(
                f"fairness.roc_auc[{name}]",
                "fairness",
                auc is not None and auc >= cfg["per_group_roc_auc_min"],
                auc,
                cfg["per_group_roc_auc_min"],
                f"support={groups['values'][name]['support']}",
            )
        )

    report.add(
        Check(
            "fairness.demographic_parity_diff",
            "fairness",
            groups["demographic_parity_diff"] <= cfg["demographic_parity_diff_max"],
            groups["demographic_parity_diff"],
            cfg["demographic_parity_diff_max"],
            "spread in selection rate across slices",
        )
    )
    report.add(
        Check(
            "fairness.equal_opportunity_diff",
            "fairness",
            groups["equal_opportunity_diff"] <= cfg["equal_opportunity_diff_max"],
            groups["equal_opportunity_diff"],
            cfg["equal_opportunity_diff_max"],
            "spread in true-positive rate across slices",
        )
    )


def evaluate_gate(sc: dict, baseline: dict | None, thresholds: dict) -> Report:
    report = Report()
    check_integrity(sc, thresholds["integrity"], report)
    # Integrity failures poison every downstream number, so stop here rather
    # than emit twelve confusing quality failures caused by one broken input.
    if report.failures:
        return report
    check_floors(sc, thresholds["floors"], report)
    check_regression(sc, baseline, thresholds["regression"], report)
    check_fairness(sc, thresholds["fairness"], report)
    return report


def to_markdown(sc: dict, baseline: dict | None, report: Report) -> str:
    m = sc["metrics"]
    base = baseline["metrics"] if baseline else None
    icon = "✅" if report.decision == "PROMOTE" else "🚫"

    lines = [
        f"## {icon} Model evaluation gate — **{report.decision}**",
        "",
        f"`{sc['model']['git_sha']}` · snapshot `{sc['model']['snapshot']}` · "
        f"run `{sc['model']['run_id'][:12]}` · {sc['integrity']['test_rows']:,} held-out rows",
        "",
        "| metric | challenger | champion | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key in ("roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier_score"):
        cur = m[key]
        if base is None:
            lines.append(f"| {key} | {cur:.4f} | — | — |")
            continue
        delta = cur - base[key]
        # brier is a loss: a rise is the bad direction.
        good = delta <= 0 if key == "brier_score" else delta >= 0
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        mark = "" if good or abs(delta) < 1e-9 else " ⚠️"
        lines.append(f"| {key} | {cur:.4f} | {base[key]:.4f} | {arrow} {delta:+.4f}{mark} |")

    g = sc["groups"]
    lines += [
        "",
        f"**Fairness** (`{g['column']}`) — parity Δ {g['demographic_parity_diff']:.3f}, "
        f"equal-opportunity Δ {g['equal_opportunity_diff']:.3f}, "
        f"worst slice `{g['worst_group']}` at {_fmt(g['worst_group_roc_auc'])}",
        "",
        "| slice | support | ROC-AUC | recall | selection rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, gv in g["values"].items():
        lines.append(
            f"| {name} | {gv['support']:,} | {_fmt(gv['roc_auc'])} | "
            f"{_fmt(gv['recall'])} | {_fmt(gv['selection_rate'])} |"
        )

    if report.failures:
        lines += [
            "",
            "### Blocking checks",
            "",
            "| check | observed | limit | why |",
            "|---|---:|---:|---|",
        ]
        for c in report.failures:
            lines.append(f"| `{c.name}` | {_fmt(c.observed)} | {_fmt(c.limit)} | {c.detail} |")
        lines += ["", "The model was **not** registered. Thresholds live in `thresholds.yaml`."]
    else:
        passed = len(report.checks)
        lines += [
            "",
            f"All {passed} checks passed — registering as `challenger` and starting shadow deploy.",
        ]

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="apply promotion thresholds to a scorecard")
    ap.add_argument("--scorecard", type=Path, default=Path("artifacts/scorecard.json"))
    ap.add_argument("--baseline", type=Path, default=Path("baselines/champion.json"))
    ap.add_argument("--thresholds", type=Path, default=Path("thresholds.yaml"))
    ap.add_argument("--json-out", type=Path, default=Path("artifacts/verdict.json"))
    ap.add_argument("--markdown", type=Path, default=Path("artifacts/report.md"))
    ap.add_argument("--no-baseline", action="store_true", help="floors only; skip the diff")
    ap.add_argument("--warn-only", action="store_true", help="report but always exit 0")
    args = ap.parse_args()

    sc = json.loads(args.scorecard.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))

    baseline = None
    if not args.no_baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    report = evaluate_gate(sc, baseline, thresholds)

    verdict = {
        "decision": report.decision,
        "git_sha": sc["model"]["git_sha"],
        "run_id": sc["model"]["run_id"],
        "snapshot": sc["model"]["snapshot"],
        "baseline": str(args.baseline) if baseline else None,
        "checks_total": len(report.checks),
        "checks_failed": len(report.failures),
        "metrics": sc["metrics"],
        "checks": [c.as_dict() for c in report.checks],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(to_markdown(sc, baseline, report), encoding="utf-8")

    passed = len(report.checks) - len(report.failures)
    print(f"[gate] {report.decision}: {passed}/{len(report.checks)} checks passed")
    for c in report.failures:
        print(f"  FAIL {c.name}: observed={_fmt(c.observed)} limit={_fmt(c.limit)} ({c.detail})")

    if report.decision == "BLOCKED" and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
