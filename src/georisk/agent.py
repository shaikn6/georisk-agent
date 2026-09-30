"""Orchestration: AOI -> scene (failover chain) -> features -> hazards (isolated) -> narrative."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from .analyst import build_facts, memo
from .aoi import bbox_from_point
from .calibration import calibrated_probabilities, outside_domain
from .confidence import Confidence, assess_confidence
from .hazards import HazardContext, HazardOutcome, default_hazards, run_hazards
from .providers import ProviderChain, build_default_chain
from .scene import Scene
from .scoring import Features, RiskScore, extract_features
from .synthetic import make_scene
from .terrain import TerrainFeatures, TerrainUnavailable, load_terrain

log = logging.getLogger(__name__)
_chain: ProviderChain | None = None


def default_chain() -> ProviderChain:
    global _chain
    if _chain is None:
        _chain = build_default_chain()
    return _chain


@dataclass
class Assessment:
    lat: float
    lon: float
    radius_m: float
    bbox: list[float]
    area_km2: float
    source: str
    acquired: str | None
    cloud_cover: float | None
    features: Features
    terrain: TerrainFeatures | None
    hazards: list[HazardOutcome]
    overall: float | None
    overall_level: str
    confidence: Confidence
    provenance: dict
    narrative: str
    narrative_source: str
    warnings: list[str] = field(default_factory=list)
    calibrated: dict = field(default_factory=dict)

    @property
    def scores(self) -> list[RiskScore]:
        return [h.score for h in self.hazards if h.score is not None]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scores"] = [asdict(s) for s in self.scores]
        return d


def _overall(scores: list[RiskScore]) -> tuple[float | None, str]:
    if not scores:
        return None, "unknown"
    vals = sorted((s.score for s in scores), reverse=True)
    v = round(min(100.0, vals[0] + 0.15 * sum(vals[1:])), 1)
    return v, "low" if v < 25 else "moderate" if v < 50 else "high" if v < 75 else "severe"


def assess(
    lat: float,
    lon: float,
    radius_m: float = 500,
    *,
    chain: ProviderChain | None = None,
    scene: Scene | None = None,
    archetype: str | None = None,
    terrain: TerrainFeatures | None = None,
    with_terrain: bool = True,
    composite: int = 1,
    days: int | None = None,
    use_llm: bool = True,
) -> Assessment:
    bbox = bbox_from_point(lat, lon, radius_m)
    warnings: list[str] = []
    for task in outside_domain(lat, lon):
        warnings.append(f"{task} heuristic score is unvalidated here: its calibrated model only covers the "
                        "regions it was trained on, so no probability is given")
    trail: list[dict] = []
    degraded_path = False

    if scene is None and archetype:
        scene = make_scene(archetype)
    if scene is None:
        c = chain or default_chain()
        result = c.load_composite(bbox, n=composite, days=days) if composite > 1 else c.load(bbox, days=days)
        scene, trail, degraded_path = result.scene, result.trail, result.degraded
        if degraded_path:
            warnings.append("imagery obtained via fallback path; see provenance")
    is_real = not scene.source.startswith("synthetic")

    if terrain is None and with_terrain and is_real:
        try:
            terrain = load_terrain(bbox)
        except TerrainUnavailable as exc:
            warnings.append(f"terrain unavailable: {exc}")
            log.warning("terrain unavailable: %s", exc)

    f = extract_features(scene)
    outcomes = run_hazards(HazardContext(f, terrain), default_hazards())
    for h in outcomes:
        if h.status == "failed":
            warnings.append(f"{h.name} could not be scored: {h.note}")
    scores = [h.score for h in outcomes if h.score is not None]
    overall, lvl = _overall(scores)
    conf = assess_confidence(scene, terrain_ok=terrain is not None, degraded_path=degraded_path)
    calibrated = calibrated_probabilities(f, terrain, lat=lat, lon=lon) if is_real else {}
    facts = build_facts(calibrated, scores=scores, overall=overall, overall_level=lvl, confidence=conf.level, acquired=scene.acquired,
                        cloud_cover=scene.cloud_cover, warnings=warnings, source=scene.source)
    narrative, narrative_source = memo(facts, use_llm=use_llm)
    return Assessment(
        lat=lat, lon=lon, radius_m=radius_m, bbox=bbox.as_list(), area_km2=round(bbox.area_km2(), 3),
        source=scene.source, acquired=scene.acquired, cloud_cover=scene.cloud_cover, features=f,
        terrain=terrain, hazards=outcomes, overall=overall, overall_level=lvl, confidence=conf,
        provenance={"imagery_trail": trail, "terrain": terrain is not None, "synthetic": not is_real,
                    "n_scenes": scene.meta.get("n_scenes", 1)},
        narrative=narrative, narrative_source=narrative_source, warnings=warnings, calibrated=calibrated,
    )
