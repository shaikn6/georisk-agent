"""Analyst memo: a plain-English risk note built only from the assessment's own numbers.

A deterministic template always works. An optional LLM rewrite is accepted only if every number it writes
appears in the facts it was given; otherwise the template is used. The model can polish words, never invent data.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re

log = logging.getLogger(__name__)
REVIEW_AT = 0.5  # probability at or above which the memo recommends a closer look
_NUM = re.compile(r"\d+(?:\.\d+)?")


def build_facts(calibrated: dict, *, scores: list | None = None, overall: float | None, overall_level: str, confidence: str, acquired: str | None,
                cloud_cover: float | None, warnings: list[str], source: str) -> dict:
    hazards = {}
    for task, v in calibrated.items():
        hazards[task] = {
            "probability_pct": round(v["probability"] * 100),
            "meaning": v["meaning"],
            "drivers": [{"factor": d["label"], "effect_points": d["effect_pp"]} for d in v.get("drivers", [])[:3]],
            "similar_sites": v.get("evidence"),
        }
    return {
        "hazards": hazards, "review_threshold_pct": round(REVIEW_AT * 100),
        "heuristic_index": None if overall is None else round(overall), "heuristic_level": overall_level,
        "heuristic_hazards": {x.name: {"score": round(x.score), "level": x.level} for x in (scores or [])},
        "confidence": confidence, "scene_date": (acquired or "")[:10] or None,
        "cloud_cover_pct": None if cloud_cover is None or math.isnan(cloud_cover) else round(cloud_cover),
        "warnings": warnings, "imagery_source": source.split(":")[0],
    }


def _points(x: float) -> str:
    return f"{abs(x):g} point" + ("" if abs(x) == 1 else "s")


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]  # keeps acronyms like NDVI intact


def template_memo(facts: dict) -> str:
    h = facts["hazards"]
    if not h:
        idx, hh = facts["heuristic_index"], facts["heuristic_hazards"]
        desc = "; ".join(f"{n} {v['level']} ({v['score']}/100)" for n, v in hh.items())
        return ("No validated model covers this site, so no probability is given. "
                + (f"Heuristic hazard levels: {desc}. " if desc else "")
                + (f"The overall heuristic index is {idx} out of 100 ({facts['heuristic_level']}), but in testing that index "
                   "was no better than chance, so treat it as description only. " if idx is not None else "")
                + "This is a screening estimate from satellite imagery, not an inspection.")
    lines = []
    head = "; ".join(f"{t} {v['probability_pct']}%" for t, v in h.items())
    lines.append(f"Screening result: {head}. These are fitted probabilities (modest skill), useful for ranking sites, not pricing them.")
    for t, v in h.items():
        if v["drivers"]:
            parts = [f"{_lower_first(d['factor'])} {'raises' if d['effect_points'] > 0 else 'lowers'} it by {_points(d['effect_points'])}"
                     for d in v["drivers"]]
            lines.append(f"What drives {t}: " + "; ".join(parts) + ".")
    for t, v in h.items():
        e = v.get("similar_sites")
        if e:
            lines.append(f"Track record for {t}: in testing, {e['observed_pct']}% of {e['n']} held-out sites scored {e['band_pct'][0]}-{e['band_pct'][1]}% "
                         f"were actually positive (95% range {e['ci95_pct'][0]}-{e['ci95_pct'][1]}%).")
    top_t, top_v = max(h.items(), key=lambda kv: kv[1]["probability_pct"])
    if top_v["probability_pct"] >= facts["review_threshold_pct"]:
        lines.append(f"Suggested use: the {top_t} figure is at or above {facts['review_threshold_pct']}%, so confirm it with an official "
                     "determination or an inspection before relying on it.")
    else:
        lines.append(f"Suggested use: no hazard reaches {facts['review_threshold_pct']}%; use the figures to compare against other sites.")
    caveats = []
    if facts["scene_date"]:
        cc = f", {facts['cloud_cover_pct']}% cloud" if facts["cloud_cover_pct"] is not None else ""
        caveats.append(f"Imagery from {facts['scene_date']}{cc}; confidence {facts['confidence']}.")
    caveats += [w[0].upper() + w[1:] + ("" if w.endswith(".") else ".") for w in facts["warnings"]]
    return " ".join(lines + caveats)


def allowed_numbers(facts: dict) -> set[str]:
    out: set[str] = set()
    for n in _NUM.findall(json.dumps(facts)):
        x = float(n)
        out |= {n, f"{x:g}", f"{x:.0f}", f"{x:.1f}"}
    return out


def verify_numbers(text: str, facts: dict) -> bool:
    ok = allowed_numbers(facts)
    return all(n in ok or f"{float(n):g}" in ok for n in _NUM.findall(text))


def _call_llm(facts: dict) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    import anthropic

    msg = anthropic.Anthropic(api_key=key, timeout=20).messages.create(
        model=os.environ.get("GEORISK_MODEL", "claude-sonnet-5"), max_tokens=350,
        system=("You write short property-risk memos for a lending analyst in plain English. Use ONLY the numbers in the JSON facts, "
                "exactly as given; never add a number, a place name or a cause that is not in the facts. Mention the main drivers and "
                "every warning. Say it is a screening estimate, not an inspection. Maximum 110 words."),
        messages=[{"role": "user", "content": json.dumps(facts)}],
    )
    return msg.content[0].text.strip()


def memo(facts: dict, use_llm: bool = False) -> tuple[str, str]:
    """(text, source): 'template', or 'llm' when the rewrite passed the number check."""
    if use_llm:
        try:
            text = _call_llm(facts)
        except Exception as exc:  # noqa: BLE001 - the memo is optional polish; never fail the assessment over it
            log.warning("LLM memo failed, using template: %s", exc)
            text = None
        if text and verify_numbers(text, facts):
            return text, "llm"
        if text:
            log.warning("LLM memo rejected: it contained numbers that are not in the facts")
    return template_memo(facts), "template"
