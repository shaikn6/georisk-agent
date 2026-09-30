"""FastAPI service: liveness, deep health, single and batch assessment with per-item isolation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import assess, default_chain
from .providers.chain import AllProvidersFailed
from .synthetic import ARCHETYPES

log = logging.getLogger(__name__)
app = FastAPI(title="georisk-agent", version="0.2.0")
_POOL = ThreadPoolExecutor(max_workers=4)


class AssessRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(default=500, ge=10, le=20_000)
    days: int | None = Field(default=None, ge=1, le=365)
    archetype: str | None = Field(default=None, description="Offline synthetic scene instead of live imagery")
    with_terrain: bool = True
    use_llm: bool = True


class BatchRequest(BaseModel):
    items: list[AssessRequest] = Field(min_length=1, max_length=25)


def _run(req: AssessRequest) -> dict:
    if req.archetype and req.archetype not in ARCHETYPES:
        raise ValueError(f"archetype must be one of {sorted(ARCHETYPES)}")
    return assess(
        req.lat, req.lon, req.radius_m, archetype=req.archetype, days=req.days,
        with_terrain=req.with_terrain, use_llm=req.use_llm,
    ).to_dict()


@app.get("/health")
def health() -> dict:
    """Liveness only: the process is up. Never touches external services."""
    return {"status": "ok"}


@app.get("/health/deep")
def health_deep() -> dict:
    """Readiness: probes each imagery provider and reports circuit state."""
    providers = default_chain().health()
    ok = any(p["reachable"] for p in providers.values())
    return {"status": "ok" if ok else "degraded", "providers": providers}


@app.post("/assess")
def post_assess(req: AssessRequest) -> dict:
    try:
        return _run(req)
    except AllProvidersFailed as e:
        raise HTTPException(503, {"error": "imagery unavailable", "trail": e.trail}) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/assess/batch")
def post_batch(batch: BatchRequest) -> dict:
    """Each item succeeds or fails on its own; one bad location never sinks the batch."""

    def one(req: AssessRequest) -> dict:
        try:
            return {"ok": True, "result": _run(req)}
        except AllProvidersFailed as e:
            return {"ok": False, "error": "imagery unavailable", "trail": e.trail}
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            log.exception("batch item failed")
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    results = list(_POOL.map(one, batch.items))
    return {"count": len(results), "succeeded": sum(r["ok"] for r in results), "results": results}
