"""Canary: give the challenger 5% of live traffic, then decide.

The decision is the point. This script ends in exactly one of three states,
and never in "a human will look at it later":

    PROMOTE   canary metrics held -> champion alias moves to the challenger
    ROLLBACK  a guard tripped -> traffic returns to 100% champion immediately
    HOLD      not enough traffic to decide -> canary stays at its current
              percentage and CD exits non-zero rather than promoting on thin
              evidence

Comparisons are champion-vs-challenger *within the same window*. Comparing a
canary against yesterday's dashboard is how seasonality gets mistaken for a
regression.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from deploy.client import ServiceClient, load_traffic


def evaluate(comparison: dict, cfg: dict) -> tuple[str, list[dict]]:
    checks: list[dict] = []

    def add(name, observed, limit, passed, note=""):
        checks.append(
            {"name": name, "observed": observed, "limit": limit, "passed": bool(passed),
             "note": note}
        )

    # Sample size is counted on the canary slice, not the window. 5% of
    # 20,000 requests is 1,000 canary requests; grading the guard against the
    # 20,000 would declare a canary healthy on a twentieth of the evidence.
    served = comparison["challenger_served"]
    if served < cfg["min_requests"]:
        add("canary_requests", served, cfg["min_requests"], False,
            f"only {served} of {comparison['requests']} requests reached the challenger")
        return "HOLD", checks
    add("canary_requests", served, cfg["min_requests"], True)

    add(
        "error_rate",
        comparison["challenger_error_rate"],
        cfg["max_error_rate"],
        comparison["challenger_error_rate"] <= cfg["max_error_rate"],
    )

    chal_p95 = comparison["challenger"]["p95_latency_ms"]
    add(
        "p95_latency_ms",
        chal_p95,
        cfg["p95_latency_ms_max"],
        chal_p95 is not None and chal_p95 <= cfg["p95_latency_ms_max"],
    )

    # The business guard, and the only check here that is not about the
    # software. `decision` is "predicted default", so the decline rate is the
    # lever: declining 3% more applicants is a revenue incident, and declining
    # 3% fewer is a credit-loss incident. Both directions roll back, because a
    # model whose live decision mix moved that far is not the model anyone
    # signed off on offline.
    champ_pos = comparison["champion"]["positive_rate"]
    chal_pos = comparison["challenger"]["positive_rate"]
    if champ_pos is None or chal_pos is None:
        add("decision_rate_shift", None, cfg["decision_rate_shift_max"], False,
            "one arm produced no decisions")
    else:
        shift = chal_pos - champ_pos
        add(
            "decision_rate_shift",
            round(shift, 6),
            cfg["decision_rate_shift_max"],
            abs(shift) <= cfg["decision_rate_shift_max"],
            f"champion declines {champ_pos:.3f} vs challenger {chal_pos:.3f}",
        )

    return ("PROMOTE" if all(c["passed"] for c in checks) else "ROLLBACK"), checks


def to_markdown(report: dict) -> str:
    icon = {"PROMOTE": "🚀", "ROLLBACK": "⏮️", "HOLD": "⏸️"}[report["decision"]]
    c = report["comparison"]
    lines = [
        f"## {icon} Canary — **{report['decision']}**",
        "",
        f"Challenger `v{report['challenger_version']}` answered "
        f"{c['challenger_served']:,} of {c['requests']:,} requests "
        f"({report['traffic_percent']}% target, {c['window_seconds']}s window).",
        "",
        "| | champion | challenger |",
        "|---|---:|---:|",
        f"| p95 latency (ms) | {c['champion']['p95_latency_ms']} | "
        f"{c['challenger']['p95_latency_ms']} |",
        f"| decline rate | {c['champion']['positive_rate']} | {c['challenger']['positive_rate']} |",
        "",
        "| guard | observed | limit | ok |",
        "|---|---:|---:|:--:|",
    ]
    for chk in report["checks"]:
        lines.append(
            f"| {chk['name']} | {chk['observed']} | {chk['limit']} | "
            f"{'✅' if chk['passed'] else '❌'} |"
        )
    lines += ["", {
        "PROMOTE": "Champion alias moved. 100% of traffic now on the new model.",
        "ROLLBACK": "Traffic returned to the champion. The challenger stays registered "
                    "for investigation; nothing was deleted.",
        "HOLD": "Canary left in place. Not enough traffic to promote or condemn.",
    }[report["decision"]]]
    return "\n".join(lines) + "\n"


def registry_cmd(*argv: str) -> int:
    cmd = [sys.executable, "-m", "src.model.registry", *argv]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="run a canary and promote or roll back")
    ap.add_argument("--target", default="inprocess", help="service URL or 'inprocess'")
    ap.add_argument("--traffic", type=Path, default=Path("traffic/replay.jsonl"))
    ap.add_argument("--thresholds", type=Path, default=Path("thresholds.yaml"))
    ap.add_argument("--shadow-report", type=Path, default=Path("artifacts/shadow_report.json"))
    ap.add_argument("--percent", type=float, default=None, help="override thresholds.canary")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", type=Path, default=Path("artifacts/canary_report.json"))
    ap.add_argument("--markdown", type=Path, default=Path("artifacts/canary_report.md"))
    ap.add_argument(
        "--no-act", action="store_true", help="decide and report, but do not touch the registry"
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))["canary"]
    percent = args.percent if args.percent is not None else cfg["traffic_percent"]

    # Shadow is a precondition, not a suggestion. A challenger that never ran
    # against real payloads has no business answering any of them.
    if args.shadow_report.exists():
        shadow = json.loads(args.shadow_report.read_text(encoding="utf-8"))
        if shadow.get("decision") != "PASS":
            raise SystemExit(
                f"[canary] shadow deploy was {shadow.get('decision')}; refusing canary"
            )
    else:
        raise SystemExit(f"[canary] no shadow report at {args.shadow_report}; run shadow first")

    records = load_traffic(args.traffic, args.limit)
    client = ServiceClient(args.target)
    try:
        ready = client.ready()
        client.set_mode("canary", percent)
        started = time.time()
        errors = 0
        for record in records:
            status, _ = client.predict(record)
            if status != 200:
                errors += 1
        comparison = client.stats()["comparison"]
        elapsed = time.time() - started

        decision, checks = evaluate(comparison, cfg)

        report = {
            "decision": decision,
            "target": args.target,
            "traffic_percent": percent,
            "champion_version": ready.get("champion_version"),
            "challenger_version": ready.get("challenger_version"),
            "replayed": len(records),
            "request_errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "comparison": comparison,
            "checks": checks,
            "actions": [],
        }

        if not args.no_act:
            if decision == "PROMOTE":
                registry_cmd("promote")
                client.set_mode("off")
                client.reload()
                report["actions"].append("registry: champion -> challenger version")
                report["actions"].append("serving: mode off (100% on the new champion)")
            elif decision == "ROLLBACK" and cfg.get("auto_rollback", True):
                # Cut traffic first, then reason about the registry. The
                # ordering matters: users stop seeing the bad model before
                # anyone looks at a dashboard.
                client.set_mode("off")
                report["actions"].append("serving: canary traffic cut to 0% immediately")
            else:
                report["actions"].append("no action: HOLD")
    finally:
        client.close()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(to_markdown(report), encoding="utf-8")

    print(
        f"[canary] {decision} at {percent}%: challenger answered "
        f"{comparison['challenger_served']} of {comparison['requests']} requests"
    )
    for c in checks:
        if not c["passed"]:
            print(f"  FAIL {c['name']}: observed={c['observed']} limit={c['limit']} {c['note']}")
    for action in report["actions"]:
        print(f"  action: {action}")

    return 0 if decision == "PROMOTE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
