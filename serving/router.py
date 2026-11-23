"""Traffic routing for shadow and canary, kept out of the HTTP layer.

Three modes, one request path:

    off      champion answers, nothing else happens
    shadow   champion answers; the challenger also scores the request and the
             two predictions are logged side by side. The caller cannot tell.
    canary   a deterministic slice of requests is *answered* by the
             challenger; the rest by the champion

The slice is a hash of the request's routing key, not a coin flip. The same
application_id therefore gets the same model on a retry, which is the
difference between a canary you can reason about and one that produces two
different decisions for one customer.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

MODES = ("off", "shadow", "canary")


def bucket(key: str, salt: str = "canary") -> float:
    """Stable [0, 1) position for a routing key."""
    digest = hashlib.blake2b(f"{salt}:{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def in_canary(key: str, percent: float) -> bool:
    if percent <= 0:
        return False
    if percent >= 100:
        return True
    return bucket(key) < (percent / 100.0)


@dataclass
class Observation:
    """One request as both models saw it."""

    key: str
    at: float
    arm: str  # which model actually answered
    champion_proba: float | None
    champion_decision: int | None
    champion_latency_ms: float | None
    challenger_proba: float | None = None
    challenger_decision: int | None = None
    challenger_latency_ms: float | None = None
    challenger_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrafficLog:
    """Bounded in-memory log of recent observations.

    Bounded on purpose: a serving process must not grow without limit because
    an operator forgot to turn shadow mode off. The durable copy is the
    JSONL sink; this deque only backs the /stats endpoint.
    """

    maxlen: int = 50_000
    _items: deque[Observation] = field(init=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=self.maxlen)

    def add(self, obs: Observation) -> None:
        with self._lock:
            self._items.append(obs)

    def snapshot(self) -> list[Observation]:
        with self._lock:
            return list(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile: the smallest value at or above rank ceil(q%·N).

    No numpy in the request path, and no interpolation -- an interpolated p95
    reports a latency no request actually had, which is a poor thing to gate a
    deploy on.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def summarize(observations: list[Observation]) -> dict[str, Any]:
    """The comparison both shadow and canary analysis are built on."""
    champ_lat = [o.champion_latency_ms for o in observations if o.champion_latency_ms is not None]
    chal_lat = [
        o.challenger_latency_ms for o in observations if o.challenger_latency_ms is not None
    ]
    scored = [
        o for o in observations if o.champion_decision is not None
        and o.challenger_decision is not None
    ]
    disagreements = sum(1 for o in scored if o.champion_decision != o.challenger_decision)
    errors = sum(1 for o in observations if o.challenger_error)

    champ_decisions = [o.champion_decision for o in observations if o.champion_decision is not None]
    chal_decisions = [
        o.challenger_decision for o in observations if o.challenger_decision is not None
    ]

    return {
        "requests": len(observations),
        "compared": len(scored),
        # Requests the challenger *answered*, as opposed to merely scored.
        # In shadow mode this is 0 by definition; in canary it is the only
        # count that reflects real exposure, so the canary's sample-size
        # guard reads this and not `requests`.
        "challenger_served": sum(1 for o in observations if o.arm == "challenger"),
        "challenger_errors": errors,
        "challenger_error_rate": round(errors / len(observations), 6) if observations else 0.0,
        "disagreement_rate": round(disagreements / len(scored), 6) if scored else 0.0,
        "champion": {
            "p50_latency_ms": _r(percentile(champ_lat, 50)),
            "p95_latency_ms": _r(percentile(champ_lat, 95)),
            "positive_rate": _r(_mean(champ_decisions)),
        },
        "challenger": {
            "p50_latency_ms": _r(percentile(chal_lat, 50)),
            "p95_latency_ms": _r(percentile(chal_lat, 95)),
            "positive_rate": _r(_mean(chal_decisions)),
        },
        "window_seconds": round(
            (max(o.at for o in observations) - min(o.at for o in observations)), 3
        )
        if observations
        else 0.0,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def _r(v: float | None, digits: int = 4) -> float | None:
    return round(v, digits) if v is not None else None
