"""Orchestration: AOI -> scene -> features -> scores -> narrative."""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass

from .aoi import BBox, bbox_from_point
from .scene import Scene
from .scoring import Features, RiskScore, extract_features, score_all
from .synthetic import make_scene

log = logging.getLogger(__name__)


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
    scores: list[RiskScore]
    overall: float
    overall_level: str
    narrative: str
    narrative_source: str

    def to_dict(self) -> dict:
        return asdict(self)


def _overall(scores: list[RiskScore]) -> float:
    # Worst single hazard dominates; the rest add a small correlated-risk uplift.
    vals = sorted((s.score for s in scores), reverse=True)
    return round(min(100.0, vals[0] + 0.15 * sum(vals[1:])), 1)


def _level(x: float) -> str:
    return "low" if x < 25 else "moderate" if x < 50 else "high" if x < 75 else "severe"


def template_narrative(a_scores: list[RiskScore], f: Features, src: str) -> str:
    top = max(a_scores, key=lambda s: s.score)
    lines = [f"Highest exposure: {top.name} ({top.level}, {top.score:.0f}/100). " + "; ".join(top.factors) + "."]
    for s in a_scores:
        if s is not top:
            lines.append(f"{s.name}: {s.level} ({s.score:.0f}/100).")
    lines.append(f"Based on {f.valid_fraction:.0%} usable pixels from {src}. Screening estimate only.")
    return " ".join(lines)


def llm_narrative(a_scores: list[RiskScore], f: Features, src: str) -> str | None:
    """Optional: Claude writes a short underwriter-style summary from the computed facts only."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic

        facts = "\n".join(f"- {s.name}: {s.score:.0f}/100 ({s.level}); " + "; ".join(s.factors) for s in a_scores)
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=os.environ.get("GEORISK_MODEL", "claude-sonnet-5"),
            max_tokens=300,
            system=(
                "You write 3-sentence physical-risk summaries for property screening. Use ONLY the "
                "numbers provided. Do not invent locations, dates, or causes. State this is a "
                "satellite screening estimate, not an inspection."
            ),
            messages=[{"role": "user", "content": f"Source: {src}\nUsable pixels: {f.valid_fraction:.0%}\n{facts}"}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:  # noqa: BLE001 - narrative is optional; never fail the assessment over it
        log.warning("LLM narrative failed, using template: %s", exc)
        return None


def assess(
    lat: float,
    lon: float,
    radius_m: float = 500,
    *,
    scene: Scene | None = None,
    archetype: str | None = None,
    days: int = 60,
    use_llm: bool = True,
) -> Assessment:
    bbox: BBox = bbox_from_point(lat, lon, radius_m)
    if scene is None:
        if archetype:
            scene = make_scene(archetype)
        else:
            from .stac import load_scene

            scene = load_scene(bbox, days=days)
    f = extract_features(scene)
    scores = score_all(f)
    overall = _overall(scores)
    text = llm_narrative(scores, f, scene.source) if use_llm else None
    return Assessment(
        lat=lat, lon=lon, radius_m=radius_m, bbox=bbox.as_list(), area_km2=round(bbox.area_km2(), 3),
        source=scene.source, acquired=scene.acquired, cloud_cover=scene.cloud_cover, features=f,
        scores=scores, overall=overall, overall_level=_level(overall),
        narrative=text or template_narrative(scores, f, scene.source),
        narrative_source="llm" if text else "template",
    )
