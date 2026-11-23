"""Inference service with built-in shadow and canary support.

Loads whatever the registry aliases point at -- never a path baked into the
image. Rolling back is therefore an alias move plus a reload, with no rebuild
and no redeploy.

Endpoints:

    POST /predict            score one application (the production path)
    GET  /healthz /readyz    liveness / model-loaded readiness
    GET  /stats              champion vs challenger over the recent window
    GET  /metrics            Prometheus exposition
    POST /admin/mode         off | shadow | canary, with a traffic percent
    POST /admin/reload       re-read the registry aliases

Shadow scoring happens inline and is wrapped: a challenger that raises must
never touch the response the caller gets. That is the entire promise of a
shadow deploy, and it is one try/except.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from serving.router import MODES, Observation, TrafficLog, in_canary, summarize
from src.model.features import as_frame
from src.model.registry import get_registry

PARAMS = yaml.safe_load(Path(os.environ.get("PARAMS", "params.yaml")).read_text(encoding="utf-8"))
SHADOW_LOG = Path(os.environ.get("SHADOW_LOG", "artifacts/shadow_log.jsonl"))
REGISTRY_BACKEND = os.environ.get("REGISTRY_BACKEND", "auto")


class Application(BaseModel):
    application_id: str = Field(..., description="routing key; also the canary bucket key")
    income: float | None = None
    debt_ratio: float | None = None
    credit_lines: int | None = None
    late_payments_2y: int | None = None
    utilization: float | None = None
    account_age_months: int | None = None
    employment: str | None = None
    region: str | None = None
    housing: str | None = None


class Prediction(BaseModel):
    application_id: str
    probability: float
    decision: int
    threshold: float
    # Which model answered. Callers log this; without it a canary incident is
    # impossible to attribute after the fact.
    served_by: str
    model_version: str | None
    latency_ms: float


class ModeRequest(BaseModel):
    mode: Literal["off", "shadow", "canary"]
    traffic_percent: float = 0.0


class LoadedModel:
    def __init__(self, alias: str, version: str, pipeline, meta: dict[str, Any]):
        self.alias = alias
        self.version = version
        self.pipeline = pipeline
        self.meta = meta
        # The threshold ships inside the model, not in this service's config.
        # params.yaml is only the fallback for a model old enough to predate
        # that rule.
        self.threshold = float(
            meta.get("decision_threshold") or PARAMS["model"]["decision_threshold"]
        )

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def score(self, records: list[dict]) -> tuple[list[float], list[int]]:
        frame = as_frame(records, PARAMS)
        proba = self.pipeline.predict_proba(frame)[:, 1]
        return [float(p) for p in proba], [int(p >= self.threshold) for p in proba]


class State:
    def __init__(self) -> None:
        self.mode = os.environ.get("SERVING_MODE", "off")
        self.traffic_percent = float(os.environ.get("CANARY_PERCENT", 0))
        self.champion: LoadedModel | None = None
        self.challenger: LoadedModel | None = None
        self.log = TrafficLog()
        self.started = time.time()
        self.counters = {"requests": 0, "champion": 0, "challenger": 0, "shadow_errors": 0}

    def load(self) -> dict:
        reg = get_registry(PARAMS["registry"]["model_name"], REGISTRY_BACKEND)
        self.champion = self._load_alias(reg, PARAMS["registry"]["champion_alias"])
        self.challenger = self._load_alias(reg, PARAMS["registry"]["challenger_alias"])
        return {
            "champion": self.champion.version if self.champion else None,
            "challenger": self.challenger.version if self.challenger else None,
        }

    @staticmethod
    def _load_alias(reg, alias: str) -> LoadedModel | None:
        version = reg.alias_version(alias)
        if version is None:
            return None
        try:
            pipeline, meta = reg.load_version(version)
        except Exception as exc:  # noqa: BLE001 - one broken alias is not an outage
            # A challenger that fails to load must not stop the champion from
            # serving. /readyz reports which aliases are live.
            print(f"[serving] could not load {alias} v{version}: {type(exc).__name__}: {exc}")
            return None
        return LoadedModel(alias, version, pipeline, meta)


state = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        loaded = state.load()
        print(f"[serving] loaded {loaded} mode={state.mode} canary={state.traffic_percent}%")
    except Exception as exc:  # noqa: BLE001 - readiness reports it; do not crash-loop
        # An unreachable registry must leave the process up and failing
        # /readyz, so the orchestrator holds traffic instead of crash-looping
        # a container that would recover on the next /admin/reload.
        print(f"[serving] startup load failed: {exc}")
    yield


app = FastAPI(title="credit-default-risk", version="1.0.0", lifespan=lifespan)


def _append_shadow_log(obs: Observation) -> None:
    try:
        SHADOW_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SHADOW_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obs.as_dict()) + "\n")
    except OSError as exc:
        # A full disk must not take down inference. Losing shadow rows is
        # recoverable; refusing traffic is not.
        print(f"[serving] shadow log write failed: {exc}")


@app.post("/predict", response_model=Prediction)
def predict(payload: Application) -> Prediction:
    if state.champion is None or not state.champion.ready:
        raise HTTPException(status_code=503, detail="no champion model loaded")

    record = payload.model_dump()
    key = payload.application_id
    state.counters["requests"] += 1

    t0 = time.perf_counter()
    champ_proba, champ_decision = state.champion.score([record])
    champ_ms = (time.perf_counter() - t0) * 1000.0

    obs = Observation(
        key=key,
        at=time.time(),
        arm="champion",
        champion_proba=champ_proba[0],
        champion_decision=champ_decision[0],
        champion_latency_ms=round(champ_ms, 4),
    )

    challenger_ready = state.challenger is not None and state.challenger.ready
    wants_challenger = state.mode in {"shadow", "canary"} and challenger_ready

    if wants_challenger:
        try:
            t1 = time.perf_counter()
            chal_proba, chal_decision = state.challenger.score([record])
            obs.challenger_latency_ms = round((time.perf_counter() - t1) * 1000.0, 4)
            obs.challenger_proba = chal_proba[0]
            obs.challenger_decision = chal_decision[0]
        except Exception as exc:  # noqa: BLE001 - the shadow must never leak
            obs.challenger_error = f"{type(exc).__name__}: {exc}"
            state.counters["shadow_errors"] += 1

    served = state.champion
    probability, decision = champ_proba[0], champ_decision[0]

    # Canary is the only mode where the challenger's answer leaves the process.
    if (
        state.mode == "canary"
        and obs.challenger_proba is not None
        and in_canary(key, state.traffic_percent)
    ):
        served = state.challenger
        probability, decision = obs.challenger_proba, obs.challenger_decision
        obs.arm = "challenger"

    state.counters["challenger" if obs.arm == "challenger" else "champion"] += 1
    state.log.add(obs)
    if wants_challenger:
        _append_shadow_log(obs)

    return Prediction(
        application_id=key,
        probability=round(probability, 6),
        decision=decision,
        threshold=served.threshold,
        served_by=obs.arm,
        model_version=served.version,
        latency_ms=round((time.perf_counter() - t0) * 1000.0, 4),
    )


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "uptime_s": round(time.time() - state.started, 1)}


@app.get("/readyz")
def readyz() -> dict:
    if state.champion is None or not state.champion.ready:
        raise HTTPException(status_code=503, detail="champion not loaded")
    return {
        "status": "ready",
        "mode": state.mode,
        "traffic_percent": state.traffic_percent,
        "champion_version": state.champion.version,
        "challenger_version": state.challenger.version if state.challenger else None,
    }


@app.get("/stats")
def stats() -> dict:
    return {
        "mode": state.mode,
        "traffic_percent": state.traffic_percent,
        "counters": state.counters,
        "comparison": summarize(state.log.snapshot()),
    }


@app.post("/admin/mode")
def set_mode(req: ModeRequest) -> dict:
    if req.mode not in MODES:
        raise HTTPException(status_code=400, detail=f"mode must be one of {MODES}")
    if req.mode == "canary" and (state.challenger is None or not state.challenger.ready):
        raise HTTPException(status_code=409, detail="cannot canary without a loaded challenger")
    state.mode = req.mode
    state.traffic_percent = req.traffic_percent if req.mode == "canary" else 0.0
    # A mode switch resets the window: comparing pre- and post-switch requests
    # in one aggregate is how a canary gets declared healthy on stale data.
    state.log.clear()
    return {"mode": state.mode, "traffic_percent": state.traffic_percent}


@app.post("/admin/reload")
def reload_models() -> dict:
    loaded = state.load()
    return {"loaded": loaded, "mode": state.mode}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus exposition. Grafana panels in monitoring/grafana read these."""
    comp = summarize(state.log.snapshot())
    lines = [
        "# HELP inference_requests_total Requests served, by arm.",
        "# TYPE inference_requests_total counter",
        f'inference_requests_total{{arm="champion"}} {state.counters["champion"]}',
        f'inference_requests_total{{arm="challenger"}} {state.counters["challenger"]}',
        "# HELP shadow_errors_total Challenger scoring failures (never user-visible).",
        "# TYPE shadow_errors_total counter",
        f"shadow_errors_total {state.counters['shadow_errors']}",
        "# HELP model_disagreement_rate Fraction of compared requests where the arms differ.",
        "# TYPE model_disagreement_rate gauge",
        f"model_disagreement_rate {comp['disagreement_rate']}",
        "# HELP inference_latency_p95_ms p95 latency over the recent window, by arm.",
        "# TYPE inference_latency_p95_ms gauge",
        f'inference_latency_p95_ms{{arm="champion"}} {comp["champion"]["p95_latency_ms"] or 0}',
        f'inference_latency_p95_ms{{arm="challenger"}} {comp["challenger"]["p95_latency_ms"] or 0}',
        "# HELP model_positive_rate Approval-side decision rate, by arm.",
        "# TYPE model_positive_rate gauge",
        f'model_positive_rate{{arm="champion"}} {comp["champion"]["positive_rate"] or 0}',
        f'model_positive_rate{{arm="challenger"}} {comp["challenger"]["positive_rate"] or 0}',
        "# HELP canary_traffic_percent Share of traffic answered by the challenger.",
        "# TYPE canary_traffic_percent gauge",
        f"canary_traffic_percent {state.traffic_percent}",
    ]
    return "\n".join(lines) + "\n"
