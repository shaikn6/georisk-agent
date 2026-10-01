import pytest

from georisk import analyst

CAL = {
    "flood": {"probability": 0.331, "meaning": "P(site lies in a FEMA Special Flood Hazard Area)",
              "drivers": [{"label": "Local relief", "effect_pp": -23.2}, {"label": "Terrain steepness", "effect_pp": 7.2},
                          {"label": "Low-lying position", "effect_pp": -1.0}]},
    "fire": {"probability": 0.608, "meaning": "P(burned before)", "drivers": [{"label": "Fuel dryness", "effect_pp": 3.3}]},
}


def facts(cal=None, warnings=None):
    return analyst.build_facts(CAL if cal is None else cal, overall=32.6, overall_level="moderate", confidence="high",
                               acquired="2026-09-25T16:32:18+00:00", cloud_cover=3.1, warnings=warnings or [], source="earth-search:S2C_1")


def test_template_states_every_probability_driver_and_caveat():
    t = analyst.template_memo(facts(warnings=["fire heuristic score is unvalidated here"]))
    for piece in ("flood 33%", "fire 61%", "local relief lowers it by 23.2 points", "terrain steepness raises it by 7.2 points", "low-lying position lowers it by 1 point.",
                  "2026-09-25", "3% cloud", "Fire heuristic score is unvalidated here."):
        assert piece in t
    assert "at or above 50%" in t  # fire 61% crosses the review threshold


def test_template_is_reassuring_only_when_nothing_crosses_the_threshold():
    low = {"flood": {"probability": 0.2, "meaning": "m", "drivers": []}}
    assert "no hazard reaches 50%" in analyst.template_memo(facts(low))


def test_no_validated_model_falls_back_to_heuristic_description_and_says_it_is_weak():
    t = analyst.template_memo(facts({}))
    assert "No validated model covers this site" in t and "no better than chance" in t


def test_number_guardrail_accepts_given_numbers_and_rejects_invented_ones():
    f = facts()
    assert analyst.verify_numbers("Flood is 33% and fire 61%; relief lowers flood by 23.2 points on 2026-09-25.", f)
    assert not analyst.verify_numbers("Flood is 33% and fire is 85%.", f)  # 85 is not in the facts
    assert not analyst.verify_numbers("Expect 4 floods a year.", f)


@pytest.mark.parametrize("reply,expected", [
    ("Flood 33%, fire 61%, relief lowers flood by 23.2 points. Screening estimate.", "llm"),
    ("Flood 33%, fire 61%, and a 12% chance of landslide.", "template"),  # invented number -> rejected
    (None, "template"),  # no key / no reply
])
def test_llm_rewrite_is_used_only_when_it_passes_the_check(monkeypatch, reply, expected):
    monkeypatch.setattr(analyst, "_call_llm", lambda f: reply)
    text, src = analyst.memo(facts(), use_llm=True)
    assert src == expected
    assert (text == reply) if expected == "llm" else ("Screening result" in text)


def test_llm_errors_never_break_the_memo(monkeypatch):
    def boom(f):
        raise RuntimeError("api down")
    monkeypatch.setattr(analyst, "_call_llm", boom)
    assert analyst.memo(facts(), use_llm=True)[1] == "template"


def test_acronyms_keep_their_case_in_driver_names():
    cal = {"flood": {"probability": 0.3, "meaning": "m", "drivers": [{"label": "Vegetation greenness (NDVI)", "effect_pp": 2.8}]}}
    assert "vegetation greenness (NDVI) raises it by 2.8 points" in analyst.template_memo(facts(cal))


SOLAR = {"probability": 0.72, "meaning": "m", "kind": "opportunity",
         "drivers": [{"label": "Flat ground (under 3 degrees)", "effect_pp": 18.4}, {"label": "Height above nearest stream", "effect_pp": -4.0}],
         "evidence": {"n": 210, "observed_pct": 74, "ci95_pct": [68, 80], "band_pct": [70, 80]},
         "caveat": "Experimental: held-out AUC 0.68 across states, but 0.63 in Ohio (only 26 test sites). Treat it as a rough screen."}


def test_memo_gives_solar_its_own_energy_paragraph_without_hazard_advice():
    f = analyst.build_facts({**CAL, "solar": SOLAR}, overall=30.0, overall_level="moderate", confidence="high", acquired=None,
                            cloud_cover=None, warnings=[], source="x")
    assert set(f["hazards"]) == {"flood", "fire"} and set(f["opportunities"]) == {"solar"}  # solar is not mixed into the hazards
    t = analyst.template_memo(f)
    assert "Energy lens: terrain similarity for solar siting is 72%" in t
    assert "flat ground (under 3 degrees) helps by 18.4 points" in t and "height above nearest stream hurts by 4 points" in t
    assert "74% of 210 sites scored 70-80% were real solar-plant sites" in t
    assert "irradiance, land price and permitting are not assessed" in t  # says what it does NOT assess (grid distance IS used)
    assert "Experimental: held-out AUC 0.68 across states, but 0.63 in Ohio (only 26 test sites)" in t  # the weakness is stated, not buried
    assert t.count("at or above 50%") == 1  # the "confirm with an official determination" advice is about hazards only
    assert analyst.verify_numbers(t, f)


def test_a_site_with_only_an_opportunity_still_gets_a_clear_memo():
    f = analyst.build_facts({"solar": SOLAR}, overall=None, overall_level="unknown", confidence="high", acquired=None,
                            cloud_cover=None, warnings=[], source="x")
    t = analyst.template_memo(f)
    assert t.startswith("Energy lens") and "not a site assessment" in t and "Screening result" not in t
