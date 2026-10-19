"""Calibration regression guard on the committed validation data and shipped artifact.

The recalibration decision (ship Platt scaling only if it improves both expected calibration error and Brier)
is made by `validation.calibrate`. These tests fail if retraining, a data change or a hand edit of the
artifact breaks that contract, or pushes the shipped probabilities' held-out calibration error past a ceiling.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from georisk.validation import calibrate
from georisk.validation.ensemble import Ensemble
from georisk.validation.metrics import brier, ece

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "validation_data/report.json"
ARTIFACT = ROOT / "src/georisk/data/calibration.json"
# Measured held-out (leave-region-out, nested) ECE of the shipped flood and solar probabilities is 0.035 and 0.022.
# The ceiling sits above both with room for harmless retraining noise, and well below the uncorrected 0.091 flood score.
ECE_CEILING = 0.05

pytestmark = pytest.mark.skipif(not REPORT.exists(), reason="committed validation data not present")


def _recal_by_task():
    return {r["task"]: r["recalibration"] for r in json.loads(REPORT.read_text()) if r.get("shipped")}


@pytest.mark.parametrize("task", ["flood", "fire", "solar"])
def test_shipped_recalibration_follows_the_gate_and_meets_the_ece_ceiling(task):
    recal, artifact = _recal_by_task()[task], json.loads(ARTIFACT.read_text())[task]
    improves_both = recal["ece_after"] < recal["ece_before"] and recal["brier_after"] <= recal["brier_before"]
    assert recal["shipped"] == improves_both
    assert ("calibrator" in artifact) == recal["shipped"]  # a correction is served iff it was shown to help
    if recal["shipped"]:
        assert recal["ece_after"] <= ECE_CEILING


def _held_out_recalibration(task, files, members, monkeypatch):
    monkeypatch.chdir(ROOT)  # load_rows joins the hydrology/topography/grid files by relative path
    rows = calibrate.load_rows(*[ROOT / f"validation_data/{f}.jsonl" for f in files])
    X, y, _ = calibrate.build_matrix(rows, calibrate.TASKS[task]["features"])
    groups = np.array([r["region"] for r in rows])
    oof, _ = calibrate._lro(lambda: Ensemble(calibrate.L2, members), X, y, groups)
    fixed = calibrate._nested_platt(oof, y, groups)
    ok = ~np.isnan(fixed)
    return {"ece_before": ece(y[ok], oof[ok]), "ece_after": ece(y[ok], fixed[ok]),
            "brier_before": brier(y[ok], oof[ok]), "brier_after": brier(y[ok], fixed[ok])}


def test_flood_platt_scaling_still_fixes_the_held_out_calibration(monkeypatch):
    pytest.importorskip("sklearn")  # the served flood blend trains its forest with scikit-learn
    got = _held_out_recalibration("flood", ["flood", "flood2", "flood3", "flood_ohio"], ("logistic", "sk_forest"), monkeypatch)
    want = _recal_by_task()["flood"]
    assert got["ece_after"] < got["ece_before"] and got["brier_after"] < got["brier_before"]
    assert got["ece_after"] <= ECE_CEILING
    for k in got:  # the committed report still describes what the data and code produce
        assert got[k] == pytest.approx(want[k], abs=2e-3)


def test_fire_platt_scaling_still_fails_the_gate_so_the_raw_probability_is_served(monkeypatch):
    got = _held_out_recalibration("fire", ["fire", "fire2"], ("logistic", "forest", "knn"), monkeypatch)
    assert got["brier_after"] > got["brier_before"]  # the correction hurts fire, which is why none is shipped
    assert "calibrator" not in json.loads(ARTIFACT.read_text())["fire"]
