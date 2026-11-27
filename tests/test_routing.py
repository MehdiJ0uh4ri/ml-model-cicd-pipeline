"""Canary routing and the shadow/canary graders.

The properties worth testing here are the ones an incident depends on: a
request always lands on the same arm, the split is close to what was asked
for, and the graders refuse to promote on thin or bad evidence.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from deploy.canary import evaluate as grade_canary
from deploy.shadow import evaluate as grade_shadow
from serving.router import Observation, TrafficLog, in_canary, percentile, summarize

THRESHOLDS = yaml.safe_load(Path("thresholds.yaml").read_text(encoding="utf-8"))
KEYS = [f"APP-{i:07d}" for i in range(20000)]


def test_routing_is_stable_across_calls():
    assert all(in_canary(k, 5) == in_canary(k, 5) for k in KEYS[:500])


def test_split_is_close_to_the_requested_percentage():
    share = sum(in_canary(k, 5) for k in KEYS) / len(KEYS)
    assert 0.045 <= share <= 0.055


def test_canary_share_is_monotonic_in_percent():
    # Ramping 5% -> 25% must keep the original 5% inside the canary, or the
    # ramp re-rolls the dice on customers who were already migrated.
    five = {k for k in KEYS[:5000] if in_canary(k, 5)}
    twentyfive = {k for k in KEYS[:5000] if in_canary(k, 25)}
    assert five <= twentyfive


def test_zero_and_full_percent_are_absolute():
    assert not any(in_canary(k, 0) for k in KEYS[:200])
    assert all(in_canary(k, 100) for k in KEYS[:200])


def test_percentile_is_nearest_rank():
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
    assert percentile([], 95) is None


def observations(n=100, disagree=0, errors=0, chal_latency=1.0, canary_percent=0.0):
    """Synthetic window. `canary_percent` marks the slice the challenger
    actually answered, which is what the canary grader counts."""
    out = []
    for i in range(n):
        out.append(
            Observation(
                key=f"k{i}",
                at=1000.0 + i,
                arm="challenger" if i < int(n * canary_percent / 100.0) else "champion",
                champion_proba=0.3,
                champion_decision=0,
                champion_latency_ms=1.0,
                challenger_proba=0.8 if i < disagree else 0.3,
                challenger_decision=1 if i < disagree else 0,
                challenger_latency_ms=chal_latency,
                challenger_error="boom" if i < errors else None,
            )
        )
    return out


def test_summary_counts_disagreements_and_errors():
    s = summarize(observations(n=200, disagree=20, errors=2))
    assert s["disagreement_rate"] == 0.1
    assert s["challenger_errors"] == 2
    assert s["compared"] == 200


def test_traffic_log_is_bounded():
    log = TrafficLog(maxlen=50)
    for obs in observations(n=200):
        log.add(obs)
    assert len(log.snapshot()) == 50


def test_shadow_blocks_a_slow_challenger():
    s = summarize(observations(n=3000, chal_latency=5.0))
    checks = grade_shadow(s, THRESHOLDS["shadow"])
    failed = {c["name"] for c in checks if not c["passed"]}
    assert "p95_latency_ratio" in failed


def test_shadow_blocks_on_mass_disagreement():
    s = summarize(observations(n=3000, disagree=900))
    failed = {c["name"] for c in grade_shadow(s, THRESHOLDS["shadow"]) if not c["passed"]}
    assert "disagreement_rate" in failed


def test_shadow_passes_a_healthy_challenger():
    s = summarize(observations(n=3000, disagree=60))
    assert all(c["passed"] for c in grade_shadow(s, THRESHOLDS["shadow"]))


def test_canary_holds_on_thin_traffic():
    thin = summarize(observations(n=2000, canary_percent=5))
    decision, _ = grade_canary(thin, THRESHOLDS["canary"])
    assert decision == "HOLD"


def test_canary_counts_the_slice_not_the_window():
    """A busy window with a tiny canary is still a tiny canary."""
    wide = summarize(observations(n=100_000, canary_percent=0.1))
    decision, checks = grade_canary(wide, THRESHOLDS["canary"])
    assert wide["requests"] == 100_000
    assert decision == "HOLD"
    hold = next(c for c in checks if c["name"] == "canary_requests")
    assert hold["observed"] == 100


def test_canary_promotes_when_arms_agree():
    decision, _ = grade_canary(
        summarize(observations(n=10_000, canary_percent=5)), THRESHOLDS["canary"]
    )
    assert decision == "PROMOTE"


def test_canary_rolls_back_on_decision_rate_shift():
    # Challenger declines 10% more applicants than the champion.
    s = summarize(observations(n=10_000, disagree=1000, canary_percent=5))
    decision, checks = grade_canary(s, THRESHOLDS["canary"])
    assert decision == "ROLLBACK"
    assert "decision_rate_shift" in {c["name"] for c in checks if not c["passed"]}


def test_decision_rate_guard_is_two_sided():
    """Approving everyone is as much an incident as declining everyone."""
    obs = observations(n=10_000, canary_percent=5)
    for o in obs:
        o.champion_decision = 1  # champion declines everyone
    s = summarize(obs)
    decision, checks = grade_canary(s, THRESHOLDS["canary"])
    shift = next(c for c in checks if c["name"] == "decision_rate_shift")
    assert shift["observed"] < 0 and not shift["passed"]
    assert decision == "ROLLBACK"


def test_canary_rolls_back_on_errors():
    s = summarize(observations(n=10_000, errors=500, canary_percent=5))
    decision, _ = grade_canary(s, THRESHOLDS["canary"])
    assert decision == "ROLLBACK"


def test_shadow_mode_serves_nothing_from_the_challenger():
    s = summarize(observations(n=1000))
    assert s["challenger_served"] == 0
    assert s["compared"] == 1000
