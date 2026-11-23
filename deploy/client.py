"""One client abstraction for the two ways a deploy step talks to the service.

    --target http://host:8080   a real running service (what CD uses)
    --target inprocess          the same FastAPI app via Starlette's TestClient

The in-process target is not a mock. It is the identical application object
with the identical routing, model loading and shadow logic, which is what
makes `make shadow` on a laptop a genuine rehearsal of the CD job rather than
a different code path that happens to print similar words.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


class ServiceClient:
    def __init__(self, target: str):
        self.target = target
        self._lifespan = None
        if target == "inprocess":
            from fastapi.testclient import TestClient

            from serving.app import app

            # Entering the context manager is what runs the app's lifespan --
            # without it the models never load and every call 503s. Held open
            # until close() so the in-process target behaves like a server
            # that is already up.
            self._client = TestClient(app)
            self._client.__enter__()
            self._lifespan = self._client
            self._base = ""
        else:
            self._client = httpx.Client(base_url=target.rstrip("/"), timeout=30.0)
            self._base = ""

    # -- plumbing ----------------------------------------------------------

    def _get(self, path: str) -> dict:
        r = self._client.get(self._base + path)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> httpx.Response:
        return self._client.post(self._base + path, json=payload)

    # -- operations --------------------------------------------------------

    def ready(self) -> dict:
        r = self._client.get(self._base + "/readyz")
        if r.status_code == 503:
            # The usual cause is an empty registry, which is a setup problem
            # rather than a deploy failure. Say so instead of surfacing a bare
            # HTTP error from three frames down.
            raise SystemExit(
                "[deploy] service has no champion loaded. For the very first model, promote it "
                "directly: `python -m src.model.registry register && "
                "python -m src.model.registry promote`. Shadow and canary apply from the "
                "second release onward."
            )
        r.raise_for_status()
        return r.json()

    def stats(self) -> dict:
        return self._get("/stats")

    def set_mode(self, mode: str, traffic_percent: float = 0.0) -> dict:
        r = self._post("/admin/mode", {"mode": mode, "traffic_percent": traffic_percent})
        if r.status_code >= 400:
            raise SystemExit(f"[deploy] mode switch rejected ({r.status_code}): {r.text}")
        return r.json()

    def reload(self) -> dict:
        r = self._post("/admin/reload", {})
        r.raise_for_status()
        return r.json()

    def predict(self, record: dict) -> tuple[int, dict | str]:
        r = self._post("/predict", record)
        try:
            return r.status_code, r.json()
        except json.JSONDecodeError:
            return r.status_code, r.text

    def close(self) -> None:
        if self._lifespan is not None:
            self._lifespan.__exit__(None, None, None)
            self._lifespan = None
            return
        close = getattr(self._client, "close", None)
        if close:
            close()


def load_traffic(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read the replay file: one JSON request body per line."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            records.append(json.loads(line))
            if limit and len(records) >= limit:
                break
    if not records:
        raise SystemExit(f"[deploy] no traffic in {path}; run `make traffic` first")
    return records
