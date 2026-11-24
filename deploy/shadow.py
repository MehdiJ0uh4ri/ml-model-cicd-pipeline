"""Shadow deploy: run the challenger on real traffic with zero user impact.

What shadow actually proves, and what it cannot:

  * proves  -- the challenger loads, accepts production payloads, does not
               raise on real feature distributions, and is not dramatically
               slower than the champion;
  * proves  -- how often it would have decided differently, which is the
               number that tells you whether the canary is worth running;
  * cannot  -- that it is *better*. Nobody sees its answers, so there is no
               outcome to compare. Quality is the eval gate's job; shadow is
               an operational check.

Conflating those two is the usual way a shadow deploy turns into theatre.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import yaml

from deploy.client import ServiceClient, load_traffic


def to_markdown(report: dict) -> str:
    c = report["comparison"]
    icon = "👻" if report["decision"] == "PASS" else "🚫"
    lines = [
        f"## {icon} Shadow deploy — **{report['decision']}**",
        "",
        f"Challenger `v{report['challenger_version']}` scored {c['requests']:,} live requests "
        f"alongside champion `v{report['champion_version']}`. "
        "No shadow output reached a caller.",
        "",
        "| | champion | challenger |",
        "|---|---:|---:|",
        f"| p50 latency (ms) | {c['champion']['p50_latency_ms']} | "
        f"{c['challenger']['p50_latency_ms']} |",
        f"| p95 latency (ms) | {c['champion']['p95_latency_ms']} | "
        f"{c['challenger']['p95_latency_ms']} |",
        f"| decline rate | {c['champion']['positive_rate']} | "
        f"{c['challenger']['positive_rate']} |",
        "",
        f"Disagreement rate **{c['disagreement_rate']:.3f}** on {c['compared']:,} compared "
        f"requests · challenger errors **{c['challenger_errors']}**",
        "",
        "| check | observed | limit | ok |",
        "|---|---:|---:|:--:|",
    ]
    for chk in report["checks"]:
        mark = "✅" if chk["passed"] else "❌"
        lines.append(f"| {chk['name']} | {chk['observed']} | {chk['limit']} | {mark} |")
    if report["decision"] != "PASS":
        lines += ["", "Canary **not** started. The challenger stays registered but unrouted."]
    return "\n".join(lines) + "\n"


def evaluate(comparison: dict, cfg: dict) -> list[dict]:
    checks = []

    def add(name, observed, limit, passed, note=""):
        checks.append(
            {"name": name, "observed": observed, "limit": limit, "passed": bool(passed),
             "note": note}
        )

    add(
        "requests",
        comparison["requests"],
        cfg["min_requests"],
        comparison["requests"] >= cfg["min_requests"],
        "too little traffic for the rates below to mean anything",
    )
    add(
        "challenger_error_rate",
        comparison["challenger_error_rate"],
        cfg["max_error_rate"],
        comparison["challenger_error_rate"] <= cfg["max_error_rate"],
    )

    champ_p95 = comparison["champion"]["p95_latency_ms"] or 0.0
    chal_p95 = comparison["challenger"]["p95_latency_ms"]
    if chal_p95 is None:
        add("p95_latency_ratio", None, cfg["p95_latency_ratio_max"], False,
            "challenger produced no timed responses")
    else:
        ratio = chal_p95 / champ_p95 if champ_p95 else 1.0
        add(
            "p95_latency_ratio",
            round(ratio, 3),
            cfg["p95_latency_ratio_max"],
            ratio <= cfg["p95_latency_ratio_max"],
        )

    add(
        "disagreement_rate",
        comparison["disagreement_rate"],
        cfg["max_disagreement_rate"],
        comparison["disagreement_rate"] <= cfg["max_disagreement_rate"],
        "a challenger that flips this many decisions needs a human before canary",
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="run a shadow deploy and grade it")
    ap.add_argument("--target", default="inprocess", help="service URL or 'inprocess'")
    ap.add_argument("--traffic", type=Path, default=Path("traffic/replay.jsonl"))
    ap.add_argument("--thresholds", type=Path, default=Path("thresholds.yaml"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json-out", type=Path, default=Path("artifacts/shadow_report.json"))
    ap.add_argument("--markdown", type=Path, default=Path("artifacts/shadow_report.md"))
    ap.add_argument("--keep-mode", action="store_true", help="leave shadow mode on when done")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))["shadow"]
    records = load_traffic(args.traffic, args.limit)

    client = ServiceClient(args.target)
    try:
        ready = client.ready()
        if not ready.get("challenger_version"):
            raise SystemExit("[shadow] no challenger registered; nothing to shadow")

        client.set_mode("shadow")
        started = time.time()
        failures = 0
        for record in records:
            status, _ = client.predict(record)
            # A non-200 here is a *champion* failure: the shadow path is
            # wrapped and cannot fail the request. It still aborts the run.
            if status != 200:
                failures += 1
                if failures > 10:
                    raise SystemExit(f"[shadow] production path failing (status {status}); aborted")
        comparison = client.stats()["comparison"]
        elapsed = time.time() - started

        checks = evaluate(comparison, cfg)
        report = {
            "decision": "PASS" if all(c["passed"] for c in checks) else "BLOCKED",
            "target": args.target,
            "champion_version": ready.get("champion_version"),
            "challenger_version": ready.get("challenger_version"),
            "replayed": len(records),
            "production_failures": failures,
            "elapsed_seconds": round(elapsed, 2),
            "comparison": comparison,
            "checks": checks,
        }
    finally:
        if not args.keep_mode:
            # Best effort: if the service is already gone, the cleanup failing
            # must not mask the shadow result the caller came for.
            with contextlib.suppress(SystemExit):
                client.set_mode("off")
        client.close()

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(to_markdown(report), encoding="utf-8")

    print(
        f"[shadow] {report['decision']}: {comparison['requests']} requests, "
        f"disagreement={comparison['disagreement_rate']:.3f}, "
        f"errors={comparison['challenger_errors']}"
    )
    for c in checks:
        if not c["passed"]:
            print(f"  FAIL {c['name']}: observed={c['observed']} limit={c['limit']} {c['note']}")

    return 0 if report["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
