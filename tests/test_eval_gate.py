"""Gate logic, tested without training anything.

The gate is the component with the most authority in the repo and the least
excuse for being slow to test: it is a pure function from (scorecard,
baseline, thresholds) to a decision.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from src.model.gate import evaluate_gate

THRESHOLDS = yaml.safe_load(Path("thresholds.yaml").read_text(encoding="utf-8"))


def scorecard(**overrides) -> dict:
    sc = {
        "model": {
            "run_id": "abc123def456",
            "git_sha": "deadbee",
            "snapshot": "2026-09-01",
            "decision_threshold": 0.42,
            "sklearn_version": "1.5.2",
            "trained_at": "2026-09-01T00:00:00Z",
        },
        "integrity": {
            "test_rows": 4800,
            "prediction_nan_rate": 0.0,
            "train_data_hash": "hash-a",
            "eval_data_hash": "hash-a",
            "data_hash_matches": True,
        },
        # Mirrors the committed champion in baselines/champion.json. Keeping
        # the fixture on the real numbers means a threshold change that would
        # block the actual champion fails here first.
        "metrics": {
            "roc_auc": 0.870,
            "pr_auc": 0.738,
            "accuracy": 0.818,
            "precision": 0.660,
            "recall": 0.687,
            "f1": 0.674,
            "brier_score": 0.123,
            "positive_rate": 0.284,
            "base_rate": 0.273,
        },
        "groups": {
            "column": "age_group",
            "values": {
                "18-25": {"support": 912, "roc_auc": 0.887, "recall": 0.712, "selection_rate": 0.31},  # noqa: E501
                "26-40": {"support": 1810, "roc_auc": 0.870, "recall": 0.687, "selection_rate": 0.28},  # noqa: E501
                "41-60": {"support": 1488, "roc_auc": 0.875, "recall": 0.681, "selection_rate": 0.27},  # noqa: E501
                "60+": {"support": 666, "roc_auc": 0.832, "recall": 0.665, "selection_rate": 0.28},
            },
            "groups_evaluated": ["18-25", "26-40", "41-60", "60+"],
            "groups_skipped_low_support": [],
            "demographic_parity_diff": 0.041,
            "equal_opportunity_diff": 0.048,
            "worst_group": "60+",
            "worst_group_roc_auc": 0.832,
        },
        "calibration": [],
    }
    for path, value in overrides.items():
        section, key = path.split(".", 1)
        sc[section][key] = value
    return sc


def baseline(**metric_overrides) -> dict:
    base = {"metrics": copy.deepcopy(scorecard()["metrics"])}
    base["metrics"].update(metric_overrides)
    base["model"] = {"git_sha": "0000000"}
    return base


def failed_names(report) -> set[str]:
    return {c.name for c in report.failures}


def test_healthy_model_promotes():
    report = evaluate_gate(scorecard(), baseline(), THRESHOLDS)
    assert report.decision == "PROMOTE"
    assert report.checks, "a passing gate must still record every check it ran"


def test_floor_blocks_even_when_beating_the_baseline():
    # Worse than the floor but better than a *very* bad champion: still blocked.
    sc = scorecard(**{"metrics.roc_auc": 0.75})
    report = evaluate_gate(sc, baseline(roc_auc=0.62), THRESHOLDS)
    assert report.decision == "BLOCKED"
    assert "floor.roc_auc" in failed_names(report)


def test_regression_blocks_a_model_that_clears_the_floor():
    sc = scorecard(**{"metrics.roc_auc": 0.845})
    report = evaluate_gate(sc, baseline(roc_auc=0.880), THRESHOLDS)
    assert report.decision == "BLOCKED"
    assert "regression.roc_auc" in failed_names(report)
    assert "floor.roc_auc" not in failed_names(report)


def test_noise_sized_drop_is_tolerated():
    sc = scorecard(**{"metrics.roc_auc": 0.866})
    report = evaluate_gate(sc, baseline(roc_auc=0.870), THRESHOLDS)
    assert report.decision == "PROMOTE"


def test_calibration_regression_blocks():
    sc = scorecard(**{"metrics.brier_score": 0.139})  # still under the floor
    report = evaluate_gate(sc, baseline(brier_score=0.120), THRESHOLDS)
    assert "regression.brier_score" in failed_names(report)


def test_broken_slice_blocks_despite_healthy_aggregate():
    sc = scorecard()
    sc["groups"]["values"]["18-25"]["roc_auc"] = 0.62
    sc["groups"]["worst_group"] = "18-25"
    sc["groups"]["worst_group_roc_auc"] = 0.62
    report = evaluate_gate(sc, baseline(), THRESHOLDS)
    assert report.decision == "BLOCKED"
    assert "fairness.roc_auc[18-25]" in failed_names(report)


def test_parity_spread_blocks():
    sc = scorecard()
    sc["groups"]["demographic_parity_diff"] = 0.18
    report = evaluate_gate(sc, baseline(), THRESHOLDS)
    assert "fairness.demographic_parity_diff" in failed_names(report)


def test_integrity_failure_short_circuits_quality_checks():
    sc = scorecard(**{"metrics.roc_auc": 0.10})
    sc["integrity"]["data_hash_matches"] = False
    report = evaluate_gate(sc, baseline(), THRESHOLDS)
    names = failed_names(report)
    assert names == {"data_hash_matches"}, "a broken measurement must not report as bad quality"


def test_short_test_set_blocks():
    sc = scorecard()
    sc["integrity"]["test_rows"] = 120
    report = evaluate_gate(sc, baseline(), THRESHOLDS)
    assert "test_rows" in failed_names(report)


def test_first_model_passes_on_floors_alone():
    report = evaluate_gate(scorecard(), None, THRESHOLDS)
    assert report.decision == "PROMOTE"
    assert "baseline.present" in {c.name for c in report.checks}


def test_single_evaluable_slice_blocks_the_fairness_gate():
    sc = scorecard()
    sc["groups"]["groups_evaluated"] = ["26-40"]
    report = evaluate_gate(sc, baseline(), THRESHOLDS)
    assert "fairness.coverage" in failed_names(report)


@pytest.mark.parametrize("metric", ["roc_auc", "pr_auc", "accuracy", "recall"])
def test_every_floor_is_wired(metric):
    sc = scorecard(**{f"metrics.{metric}": 0.01})
    report = evaluate_gate(sc, None, THRESHOLDS)
    assert f"floor.{metric}" in failed_names(report)
