"""The tooling that produces the shipped models. A bug here ships a bad model, so it is tested like served code."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from georisk.providers.chain import ChainResult
from georisk.scene import BANDS, Scene
from georisk.scoring import Features
from georisk.terrain import TerrainFeatures, TerrainUnavailable
from georisk.validation import calibrate, dataset, hydro_collect, labels

FLOOD_COLS = calibrate.TASKS["flood"]["features"]


def synth_rows(n=240, signal=True, seed=0, regions=("a", "b", "c", "d", "e", "f")):
    """Rows shaped like collected data. With signal=True, water_fraction + low relief drive the label."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cols = set(FLOOD_COLS) | set(Features.__dataclass_fields__) | {"t_mean_slope_deg", "t_p90_slope_deg", "t_relief_m", "t_center_percentile", "t_low_lying_fraction"}
        r = {c: float(rng.random()) for c in cols}
        r |= {"lat": 30 + i * 0.001, "lon": -90 - i * 0.001, "region": regions[i % len(regions)], "valid_fraction": 1.0, "task": "flood"}
        r["label"] = int(r["water_fraction"] + (1 - r["t_relief_m"]) + 0.3 * rng.normal() > 1.0) if signal else int(rng.random() < 0.5)
        rows.append(r)
    return rows


def write(path: Path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_load_rows_filters_dedupes_and_joins_hydrology(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rows = [{k: v for k, v in r.items() if not k.startswith("h_")} for r in synth_rows(10)]  # collected rows carry no river-height yet
    rows.append(dict(rows[0]))  # duplicate coordinates
    rows.append({**rows[1], "lat": 1.0, "terrain_error": "dem down"})
    rows.append({**rows[2], "lat": 2.0, "valid_fraction": 0.2})  # too cloudy
    write(tmp_path / "f.jsonl", rows)
    (tmp_path / "validation_data").mkdir()
    write(tmp_path / "validation_data/hydro.jsonl", [{"lat": rows[3]["lat"], "lon": rows[3]["lon"], "h_hand_mean_m": 4.2},
                                                    {"lat": rows[4]["lat"], "lon": rows[4]["lon"], "hydro_error": "no dem"}])
    out = calibrate.load_rows(tmp_path / "f.jsonl", tmp_path / "missing.jsonl")
    assert len(out) == 10  # duplicate, terrain-error and too-cloudy rows are gone; a missing file is ignored
    joined = {r["lat"]: r for r in out}
    assert joined[rows[3]["lat"]]["h_hand_mean_m"] == 4.2 and "h_hand_mean_m" not in joined[rows[4]["lat"]]


def test_build_matrix_fills_missing_values_with_the_median():
    rows = [{"a": 1.0, "label": 0}, {"a": 3.0, "label": 1}, {"a": None, "label": 1}, {"label": 0}]
    X, y, fill = calibrate.build_matrix(rows, ["a"])
    assert fill == {"a": 2.0} and X[:, 0].tolist() == [1.0, 3.0, 2.0, 2.0] and y.tolist() == [0, 1, 1, 0]


def test_a_real_signal_is_shipped_with_everything_the_service_needs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "f.jsonl", synth_rows(300, signal=True))
    rep = calibrate.evaluate("flood", tmp_path / "f.jsonl")
    assert rep["shipped"] and rep["served_model"] in rep["leave_region_out_by_model"]
    art = rep["artifact"]
    assert set(art) >= {"features", "fill", "model", "meaning", "domain", "reliability"} and art["features"] == FLOOD_COLS
    assert rep["model_leave_region_out"]["auc"] > 0.7 and len(art["domain"]) == 4
    sel = rep["selection"]  # the serving rule: best Brier among candidates within 0.01 AUC of the best
    lro = rep["leave_region_out_by_model"]
    assert all(lro[n]["auc"] >= sel["best_auc"] - 0.01 for n in sel["eligible"]) and rep["served_model"] in sel["eligible"]
    assert sel["final_brier"][rep["served_model"]] == min(sel["final_brier"][n] for n in sel["eligible"])
    assert all(b["n"] > 0 and b["ci"][0] <= b["rate"] <= b["ci"][1] for b in art["reliability"])


def test_pure_noise_is_refused_by_the_gate(tmp_path, monkeypatch):
    """The most important property: with no real signal, nothing ships (no fake probabilities)."""
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "f.jsonl", synth_rows(300, signal=False, seed=3))
    rep = calibrate.evaluate("flood", tmp_path / "f.jsonl")
    assert rep["shipped"] is False and "artifact" not in rep


def test_too_few_samples_never_ship(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "f.jsonl", synth_rows(20))
    assert calibrate.evaluate("flood", tmp_path / "f.jsonl") == {"task": "flood", "n": 20, "shipped": False, "reason": "too few usable samples"}


def test_domain_is_the_padded_bounding_box_of_the_training_points():
    rows = [{"lat": 30.0, "lon": -90.0}, {"lat": 40.0, "lon": -80.0}]
    assert calibrate._domain(rows, pad=1.0) == [29.0, 41.0, -91.0, -79.0]


def test_sampler_balances_classes_drops_unmapped_points_and_survives_lookup_errors(monkeypatch):
    seq = iter([1, None, 0, RuntimeError("fema down"), 1, 0, 0, 1] * 400)

    def fake_label(lat, lon):
        v = next(seq)
        if isinstance(v, Exception):
            raise v
        return v
    monkeypatch.setattr(dataset, "flood_label", fake_label)
    pts = dataset.sample_labelled_points("flood", per_class=5, seed=1)
    assert sum(p["label"] for p in pts) == 5 and len(pts) == 10
    assert {p["region"] for p in pts} <= set(dataset.FLOOD_REGIONS)
    ohio = dataset.sample_labelled_points("flood", per_class=3, seed=2, only=dataset.OHIO)
    assert {p["region"] for p in ohio} <= dataset.OHIO


def test_flood_label_reads_the_sfha_flag(monkeypatch):
    reply = {}
    monkeypatch.setattr(labels, "_get", lambda url, params, timeout=25.0: reply)
    assert labels.flood_label(1, 1) is None  # unmapped
    reply["features"] = [{"attributes": {"SFHA_TF": "F"}}]
    assert labels.flood_label(1, 1) == 0
    reply["features"] = [{"attributes": {"SFHA_TF": "F"}}, {"attributes": {"SFHA_TF": "T"}}]
    assert labels.flood_label(1, 1) == 1


def test_fire_label_distinguishes_burned_unburned_and_ambiguous(monkeypatch):
    counts = {"burned": 0, "anywhere": 0}
    monkeypatch.setattr(labels, "_fire_count", lambda where, lat, lon, pad_deg=0.0: counts["anywhere" if pad_deg else "burned"])
    assert labels.fire_label(1, 1) == 0  # nothing burned and no perimeter of any kind nearby: a clean negative
    counts["anywhere"] = 2
    assert labels.fire_label(1, 1) is None  # a small/old fire nearby: ambiguous, dropped from the data
    counts["burned"] = 1
    assert labels.fire_label(1, 1) == 1


def test_label_service_errors_are_not_retried_forever(monkeypatch):
    calls = {"n": 0}

    class R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            calls["n"] += 1
            return b'{"error": {"message": "bad query"}}'
    monkeypatch.setattr(labels.urllib.request, "urlopen", lambda *a, **k: R())
    with pytest.raises(labels.LabelUnavailable, match="bad query"):
        labels._get("http://x", {})
    assert calls["n"] == 1  # a definite service error is final, not retried


def test_feature_row_records_terrain_failures_instead_of_dropping_the_sample(monkeypatch):
    scene = Scene(bands={b: np.full((8, 8), 0.2) for b in BANDS}, valid=np.ones((8, 8), bool), source="earth-search:x",
                  acquired="2026-09-25T00:00:00+00:00", cloud_cover=1.0)
    chain = SimpleNamespace(load=lambda bbox: ChainResult(scene, []))
    monkeypatch.setattr(dataset, "load_terrain", lambda bbox: TerrainFeatures(1, 2, 3, 0.5, 0.1))
    row = dataset.feature_row({"task": "flood", "lat": 1.0, "lon": 2.0, "label": 1, "region": "x"}, chain)
    assert row["t_relief_m"] == 3 and row["label"] == 1 and "mean_ndvi" in row

    def down(bbox):
        raise TerrainUnavailable("dem down")
    monkeypatch.setattr(dataset, "load_terrain", down)
    assert "dem down" in dataset.feature_row({"task": "flood", "lat": 1.0, "lon": 2.0, "label": 0, "region": "x"}, chain)["terrain_error"]


def test_collect_is_resumable_and_isolates_failed_samples(tmp_path, monkeypatch):
    out = tmp_path / "rows.jsonl"
    pts = [{"task": "flood", "lat": float(i), "lon": 0.0, "label": 0, "region": "r"} for i in range(4)]

    def fake(pt, chain, **kw):
        if pt["lat"] == 2.0:
            raise RuntimeError("provider down")
        return {**pt, "x": 1}
    monkeypatch.setattr(dataset, "feature_row", fake)
    assert dataset.collect(pts, None, out, workers=1) == 3  # one lost sample, not a lost run
    assert dataset.collect(pts, None, out, workers=1) == 0  # re-run: the 3 done points are skipped, the failing one fails again
    assert len(out.read_text().splitlines()) == 3
    monkeypatch.setattr(dataset, "feature_row", lambda pt, chain, **kw: {**pt, "x": 2})
    again = dataset.collect(pts, None, out, workers=1)
    assert again == 1  # only the previously failed point is retried


def test_hydro_collect_points_dedupe_and_failures_are_recorded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "validation_data").mkdir()
    for n in ("flood", "flood2", "flood3", "flood_ohio"):
        write(tmp_path / f"validation_data/{n}.jsonl", [{"lat": 1.0, "lon": 2.0}, {"lat": 3.0, "lon": 4.0}])
    monkeypatch.setattr(hydro_collect, "FILES", [f"validation_data/{n}.jsonl" for n in ("flood", "flood2", "flood3", "flood_ohio")])
    assert hydro_collect.points() == [(1.0, 2.0), (3.0, 4.0)]
    hf = SimpleNamespace(hand_mean_m=1.0, hand_min_m=0.0, near_stream_fraction=0.5, dist_to_stream_m=10.0, log_stream_area=0.1, hand_big_mean_m=2.0)
    import dataclasses

    hydro = dataclasses.make_dataclass("H", [(k, float) for k in vars(hf)])(**vars(hf))
    monkeypatch.setattr(hydro_collect, "load_hydrology", lambda lat, lon: hydro)
    assert hydro_collect.one((1.0, 2.0))["h_hand_mean_m"] == 1.0

    def down(lat, lon):
        raise TerrainUnavailable("no dem")
    monkeypatch.setattr(hydro_collect, "load_hydrology", down)
    assert hydro_collect.one((1.0, 2.0))["hydro_error"] == "no dem"


def test_solar_negatives_are_nearby_but_never_on_or_beside_a_plant():
    import numpy as np

    from georisk.validation import solar

    plants = [{"lat": 35.0 + 0.3 * i, "lon": -100.0 + 0.3 * i, "state": "TX" if i % 2 else "OK", "mw": 5.0, "year": "2020"} for i in range(40)]
    pts = solar.sample_points(plants, 30, seed=2)
    assert sum(p["label"] for p in pts) == 30 and 24 <= len(pts) - 30 <= 30  # one negative per plant (a few may fail the exclusion test)
    lats, lons = np.array([p["lat"] for p in plants]), np.array([p["lon"] for p in plants])
    for p in pts:
        d = solar.haversine_km(p["lat"], p["lon"], lats, lons).min()
        assert (d < 0.01) if p["label"] == 1 else (d >= solar.EXCLUDE_KM)  # positives sit on a plant; negatives keep 2 km clear of every plant


def test_haversine_and_offset_agree_with_known_distances():
    import numpy as np

    from georisk.validation import solar

    assert solar.haversine_km(0, 0, np.array([0.0]), np.array([1.0]))[0] == pytest.approx(111.19, abs=0.1)  # one degree on the equator
    la, lo = solar.offset(40.0, -83.0, 10.0, 90.0)  # 10 km due east
    assert la == pytest.approx(40.0, abs=1e-6) and solar.haversine_km(40.0, -83.0, np.array([la]), np.array([lo]))[0] == pytest.approx(10.0, abs=0.05)


def test_plant_loader_filters_small_missing_and_out_of_area(tmp_path):
    from georisk.validation import solar

    csvp = tmp_path / "p.csv"
    csvp.write_text("ylat,xlong,p_cap_ac,p_state,p_year\n40.0,-83.0,5,OH,2020\n40.0,-83.0,0.5,OH,2020\n40.0,-83.0,,OH,2020\n"
                    "64.0,-150.0,50,AK,2020\n21.0,-157.0,50,HI,2020\nbad,-83.0,5,OH,2020\n")
    plants = solar.load_plants(csvp)
    assert [(p["state"], p["mw"]) for p in plants] == [("OH", 5.0)]  # sub-MW, missing capacity, AK/HI and unparseable rows are dropped


def test_dem_collect_records_failures_and_is_resumable(tmp_path, monkeypatch):
    from georisk.validation import dem_collect

    calls = []

    def fake(pt):
        calls.append(pt["lat"])
        return {**pt, "t_relief_m": 1.0} if pt["lat"] != 2.0 else {**pt, "terrain_error": "dem down"}
    monkeypatch.setattr(dem_collect, "one", fake)
    pts = [{"lat": float(i), "lon": 0.0, "label": i % 2} for i in range(4)]
    out = tmp_path / "s.jsonl"
    assert dem_collect.collect(pts, out, workers=1, progress=False) == 3  # one failure recorded, three good rows
    calls.clear()
    assert dem_collect.collect(pts, out, workers=1, progress=False) == 0 and calls == [2.0]  # only the failed point is retried (and fails again)
    monkeypatch.setattr(dem_collect, "one", lambda pt: {**pt, "t_relief_m": 1.0})
    assert dem_collect.collect(pts, out, workers=1, progress=False) == 1  # it succeeds this time


def test_a_region_where_the_model_tests_poorly_is_flagged_with_its_geometry(tmp_path, monkeypatch):
    """Plant a signal everywhere except 'miami', whose labels are pure noise: it must come out as a weak region."""
    monkeypatch.chdir(tmp_path)
    rows = synth_rows(360, signal=True, seed=4, regions=("houston", "dallas", "miami", "toledo", "columbus", "cleveland"))
    rng = np.random.default_rng(8)
    for r in rows:
        if r["region"] == "miami":
            r["label"] = int(rng.random() < 0.5)
    write(tmp_path / "f.jsonl", rows)
    rep = calibrate.evaluate("flood", tmp_path / "f.jsonl")
    weak = {w["name"]: w for w in rep["artifact"]["weak_regions"]}
    assert "miami" in weak and weak["miami"]["shape"] == "circle" and len(weak["miami"]["geometry"]) == 3 and weak["miami"]["n"] >= 25
    assert "houston" not in weak  # regions with real signal are not flagged
    assert rep["model_leave_region_out"]["per_region_n"]["miami"] >= 25  # per-region accuracy now comes from the served model
