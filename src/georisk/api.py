"""FastAPI service: liveness, deep health, single and batch assessment with per-item isolation."""

from __future__ import annotations

import hmac
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import resources

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .agent import assess, default_chain
from .providers.chain import AllProvidersFailed
from .synthetic import ARCHETYPES

log = logging.getLogger(__name__)
app = FastAPI(title="georisk-agent", version="0.3.0")
_POOL = ThreadPoolExecutor(max_workers=4)
# Bound total concurrent imagery work: a burst of large requests must degrade to 429, not exhaust the host.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("GEORISK_MAX_INFLIGHT", "8")))


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    """If GEORISK_API_KEY is set, every assessment call must present it (constant-time compare)."""
    expected = os.environ.get("GEORISK_API_KEY")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(401, "missing or invalid X-API-Key")


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Report which fields failed without echoing raw input (NaN/Infinity are not JSON-serialisable)."""
    fields = [{"field": ".".join(str(p) for p in e["loc"]), "problem": e["msg"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "invalid request", "details": fields})


class AssessRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(default=500, ge=10, le=20_000)
    days: int | None = Field(default=None, ge=1, le=365)
    archetype: str | None = Field(default=None, description="Offline synthetic scene instead of live imagery")
    with_terrain: bool = True
    composite: int = Field(default=1, ge=1, le=5, description="median-composite this many clear scenes")
    use_llm: bool = Field(default=False, description="LLM narrative; only honoured if GEORISK_ALLOW_LLM=1")


class BatchRequest(BaseModel):
    items: list[AssessRequest] = Field(min_length=1, max_length=25)


def _run(req: AssessRequest) -> dict:
    if req.archetype and req.archetype not in ARCHETYPES:
        raise ValueError(f"archetype must be one of {sorted(ARCHETYPES)}")
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(429, "server busy, retry shortly")
    try:
        return assess(
            req.lat, req.lon, req.radius_m, archetype=req.archetype, days=req.days,
            with_terrain=req.with_terrain, composite=req.composite,
            use_llm=req.use_llm and os.environ.get("GEORISK_ALLOW_LLM") == "1",
        ).to_dict()
    finally:
        _SLOTS.release()


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


@app.post("/assess", dependencies=[Depends(require_key)])
def post_assess(req: AssessRequest) -> dict:
    try:
        return _run(req)
    except AllProvidersFailed as e:
        raise HTTPException(503, {"error": "imagery unavailable", "trail": e.trail}) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/assess/batch", dependencies=[Depends(require_key)])
def post_batch(batch: BatchRequest) -> dict:
    """Each item succeeds or fails on its own; one bad location never sinks the batch."""

    def one(req: AssessRequest) -> dict:
        try:
            return {"ok": True, "result": _run(req)}
        except AllProvidersFailed as e:
            return {"ok": False, "error": "imagery unavailable", "trail": e.trail}
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except HTTPException as e:
            return {"ok": False, "error": str(e.detail)}
        except Exception:
            log.exception("batch item failed")  # detail stays in server logs, never in the response
            return {"ok": False, "error": "internal error"}

    results = list(_POOL.map(one, batch.items))
    return {"count": len(results), "succeeded": sum(r["ok"] for r in results), "results": results}


_CSP = ("default-src 'none'; script-src 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "style-src 'unsafe-inline' https://cdnjs.cloudflare.com; img-src 'self' data: https://tile.openstreetmap.org "
        "https://cdnjs.cloudflare.com; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo() -> HTMLResponse:
    html = resources.files("georisk").joinpath("web/index.html").read_text()
    return HTMLResponse(html, headers={"Content-Security-Policy": _CSP, "X-Content-Type-Options": "nosniff",
                                       "Referrer-Policy": "no-referrer"})
