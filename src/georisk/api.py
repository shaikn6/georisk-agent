"""FastAPI service."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import assess
from .stac import NoSceneFound
from .synthetic import ARCHETYPES

app = FastAPI(title="georisk-agent", version="0.1.0")


class AssessRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(default=500, ge=10, le=20_000)
    days: int = Field(default=60, ge=1, le=365)
    archetype: str | None = Field(default=None, description="Offline synthetic scene instead of live imagery")
    use_llm: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/assess")
def post_assess(req: AssessRequest) -> dict:
    if req.archetype and req.archetype not in ARCHETYPES:
        raise HTTPException(422, f"archetype must be one of {sorted(ARCHETYPES)}")
    try:
        return assess(req.lat, req.lon, req.radius_m, archetype=req.archetype, days=req.days,
                      use_llm=req.use_llm).to_dict()
    except NoSceneFound as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
