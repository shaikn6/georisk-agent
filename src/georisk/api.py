"""FastAPI service: liveness, deep health, single and batch assessment with per-item isolation."""

from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from importlib import resources

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .agent import default_chain
from .area import AreaError, assess_area, parse_geometry, plan_tiles
from .providers.chain import AllProvidersFailed
from .resultcache import CACHE
from .resultcache import cached_assess as assess
from .synthetic import ARCHETYPES

log = logging.getLogger(__name__)
app = FastAPI(title="georisk-agent", version="0.4.0")
_POOL = ThreadPoolExecutor(max_workers=4)
# Bound total concurrent imagery work: a burst of large requests must degrade to 429, not exhaust the host.
_SLOTS = threading.BoundedSemaphore(int(os.environ.get("GEORISK_MAX_INFLIGHT", "8")))


_HITS: dict[str, deque] = defaultdict(deque)
_HITS_LOCK = threading.Lock()


def client_ip(request: Request) -> str:
    """Only trust X-Forwarded-For when explicitly behind a proxy that sets it (GEORISK_TRUST_PROXY=1)."""
    if os.environ.get("GEORISK_TRUST_PROXY") == "1":
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def take(ip: str, cost: int = 1, *, now: float | None = None) -> bool:
    """Sliding one-minute window per client. Cost = number of assessments requested."""
    limit, t = int(os.environ.get("GEORISK_RATE_PER_MIN", "20")), time.monotonic() if now is None else now
    with _HITS_LOCK:
        q = _HITS[ip]
        while q and t - q[0] > 60:
            q.popleft()
        if len(q) + cost > limit:
            return False
        q.extend([t] * cost)
        if len(_HITS) > 10_000:  # bound memory under a spray of distinct clients
            for k in [k for k, v in _HITS.items() if not v][:5_000]:
                del _HITS[k]
        return True


def limit_rate(request: Request) -> None:
    if not take(client_ip(request)):
        raise HTTPException(429, "rate limit exceeded")


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


class AreaRequest(BaseModel):
    geometry: dict = Field(description="GeoJSON Polygon/MultiPolygon (or a one-feature Feature/FeatureCollection), WGS84")
    tile_m: float = Field(default=1000, ge=250, le=5000, description="tile side in metres")


class BatchRequest(BaseModel):
    items: list[AssessRequest] = Field(min_length=1, max_length=25)


def _run(req: AssessRequest) -> dict:
    if req.archetype and req.archetype not in ARCHETYPES:
        raise ValueError(f"archetype must be one of {sorted(ARCHETYPES)}")
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(429, "server busy, retry shortly")
    try:
        t0 = time.perf_counter()
        out = assess(
            req.lat, req.lon, req.radius_m, archetype=req.archetype, days=req.days,
            with_terrain=req.with_terrain, composite=req.composite,
            use_llm=req.use_llm and os.environ.get("GEORISK_ALLOW_LLM") == "1",
        ).to_dict()
        log.info(json.dumps({"event": "assessment", "source": str(out.get("source", ""))[:12],
                             "composite": req.composite, "degraded": bool(out["warnings"]),
                             "confidence": out["confidence"]["level"], "calibrated": sorted(out["calibrated"]),
                             "duration_ms": round((time.perf_counter() - t0) * 1000)}))
        return out
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


@app.post("/assess", dependencies=[Depends(require_key), Depends(limit_rate)])
def post_assess(req: AssessRequest) -> dict:
    try:
        return _run(req)
    except AllProvidersFailed as e:
        raise HTTPException(503, {"error": "imagery unavailable", "trail": e.trail}) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/assess/area", dependencies=[Depends(require_key)])
def post_area(req: AreaRequest, request: Request) -> dict:
    """Cut an outline into tiles and assess each one independently (a failed tile never sinks the rest)."""
    max_tiles = int(os.environ.get("GEORISK_MAX_AREA_TILES", "16"))
    try:
        tiles = plan_tiles(parse_geometry(req.geometry), req.tile_m, max_tiles)
    except AreaError as e:
        raise HTTPException(422, str(e)) from e
    if not take(client_ip(request), cost=len(tiles)):
        raise HTTPException(429, "rate limit exceeded")
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(429, "server busy, retry shortly")
    try:
        return assess_area(req.geometry, tile_m=req.tile_m, max_tiles=max_tiles)
    finally:
        _SLOTS.release()


@app.post("/assess/batch", dependencies=[Depends(require_key)])
def post_batch(batch: BatchRequest, request: Request) -> dict:
    """Each item succeeds or fails on its own; one bad location never sinks the batch."""
    if not take(client_ip(request), cost=len(batch.items)):
        raise HTTPException(429, "rate limit exceeded")

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
        "style-src 'unsafe-inline' https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data: https://services.arcgisonline.com https://cdnjs.cloudflare.com; "
        "connect-src 'self' https://nominatim.openstreetmap.org; base-uri 'none'; frame-ancestors 'none'")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo() -> HTMLResponse:
    html = resources.files("georisk").joinpath("web/index.html").read_text()
    return HTMLResponse(html, headers={"Cache-Control": "no-cache", "Content-Security-Policy": _CSP,  # revalidate: a deploy must not leave users on a stale page "X-Content-Type-Options": "nosniff",
                                       "Referrer-Policy": "no-referrer"})


_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 20, 40, 60)
_METRICS_LOCK = threading.Lock()
_REQS: dict[tuple[str, str], int] = defaultdict(int)  # (route template, status class) -> count
_LAT: dict[str, list[int]] = defaultdict(lambda: [0] * (len(_BUCKETS) + 1))  # route -> per-bucket counts
_LAT_SUM: dict[str, float] = defaultdict(float)


def _route_of(request: Request) -> str:
    r = request.scope.get("route")
    return getattr(r, "path", "unmatched")  # template, never the raw URL: bounded label set


def _record(route: str, status: int, seconds: float) -> None:
    with _METRICS_LOCK:
        _REQS[(route, f"{status // 100}xx")] += 1
        _LAT_SUM[route] += seconds
        _LAT[route][next((i for i, b in enumerate(_BUCKETS) if seconds <= b), len(_BUCKETS))] += 1


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = request.headers.get("x-request-id", "")[:64] if request.headers.get("x-request-id", "").isascii() else ""
    rid = rid or uuid.uuid4().hex[:16]
    t0 = time.perf_counter()
    status = 500
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception as exc:
        # Type and message only (tracebacks can carry request data); the client still gets a generic 500.
        log.error(json.dumps({"event": "unhandled_exception", "request_id": rid, "type": type(exc).__name__,
                              "message": str(exc)[:300]}))
        raise
    finally:
        dur = time.perf_counter() - t0
        route = _route_of(request)
        _record(route, status, dur)
        # Deliberately no coordinates, IPs or bodies: telemetry pipelines are a leak path.
        log.info(json.dumps({"event": "http_request", "request_id": rid, "method": request.method, "route": route,
                             "status": status, "duration_ms": round(dur * 1000)}))
    resp.headers["X-Request-ID"] = rid
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Strict-Transport-Security", "max-age=63072000; includeSubDomains")
    if request.url.path.startswith("/assess"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/metrics", response_class=PlainTextResponse, dependencies=[Depends(require_key)], include_in_schema=False)
def metrics() -> str:
    """Prometheus text format: request counts by route/status class and latency histograms."""
    lines = ["# TYPE georisk_result_cache_total counter",
             f'georisk_result_cache_total{{outcome="hit"}} {CACHE.hits}', f'georisk_result_cache_total{{outcome="miss"}} {CACHE.misses}',
             "# TYPE georisk_requests_total counter"]
    with _METRICS_LOCK:
        lines += [f'georisk_requests_total{{route="{r}",status_class="{c}"}} {n}' for (r, c), n in sorted(_REQS.items())]
        lines.append("# TYPE georisk_request_duration_seconds histogram")
        for route, counts in sorted(_LAT.items()):
            cum = 0
            for b, n in zip(_BUCKETS, counts, strict=False):
                cum += n
                lines.append(f'georisk_request_duration_seconds_bucket{{route="{route}",le="{b}"}} {cum}')
            total = cum + counts[-1]
            lines.append(f'georisk_request_duration_seconds_bucket{{route="{route}",le="+Inf"}} {total}')
            lines.append(f'georisk_request_duration_seconds_sum{{route="{route}"}} {_LAT_SUM[route]:.3f}')
            lines.append(f'georisk_request_duration_seconds_count{{route="{route}"}} {total}')
    return "\n".join(lines) + "\n"
