from datetime import datetime, timedelta, timezone

from georisk.confidence import assess_confidence
from georisk.synthetic import make_scene

NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


def real(days_old=3, cloud=2.0, valid=None):
    s = make_scene("healthy_forest")
    s.source = "earth-search:x"
    s.acquired = (NOW - timedelta(days=days_old)).isoformat()
    s.cloud_cover = cloud
    if valid is not None:
        s.valid[:] = False
        s.valid.flat[: int(s.valid.size * valid)] = True
    return s


def test_synthetic_has_zero_confidence():
    c = assess_confidence(make_scene("floodplain"), terrain_ok=True, degraded_path=False, now=NOW)
    assert c.score == 0.0 and c.level == "none"


def test_clear_recent_scene_with_terrain_is_high():
    c = assess_confidence(real(), terrain_ok=True, degraded_path=False, now=NOW)
    assert c.level == "high" and c.score >= 0.95


def test_each_weakness_lowers_confidence():
    base = assess_confidence(real(), terrain_ok=True, degraded_path=False, now=NOW).score
    assert assess_confidence(real(), terrain_ok=False, degraded_path=False, now=NOW).score < base
    assert assess_confidence(real(), terrain_ok=True, degraded_path=True, now=NOW).score < base
    assert assess_confidence(real(days_old=50), terrain_ok=True, degraded_path=False, now=NOW).score < base
    assert assess_confidence(real(valid=0.5), terrain_ok=True, degraded_path=False, now=NOW).score < base


def test_confidence_never_leaves_0_1():
    s = real(days_old=200, cloud=95, valid=0.1)
    s.meta["cache"] = "stale"
    c = assess_confidence(s, terrain_ok=False, degraded_path=True, now=NOW)
    assert 0.0 <= c.score <= 1.0 and c.level == "low" and len(c.reasons) >= 4
