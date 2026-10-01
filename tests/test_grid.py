import json

import numpy as np
import pytest

from georisk import grid


def line(kv, lon0, lat0, lon1, lat1):
    return kv, np.array([[lon0, lat0], [lon1, lat1]])


SITE = (40.0, -83.0)


def km_east(km):  # degrees of longitude for km at the site latitude
    return km / (111.32 * np.cos(np.radians(40.0)))


def test_distance_to_a_north_south_line_matches_its_known_offset():
    f = grid.features_from_lines(*SITE, [line(115, SITE[1] + km_east(9), 39.5, SITE[1] + km_east(9), 40.5)])
    assert f.dist_km == pytest.approx(9.0, abs=0.05) and f.n_lines_10km == 1.0 and f.max_kv_10km == 115


def test_site_on_the_line_has_zero_distance_and_the_230kv_distance_ignores_lower_voltages():
    lines = [line(115, SITE[1], 39.5, SITE[1], 40.5), line(345, SITE[1] + km_east(7), 39.5, SITE[1] + km_east(7), 40.5)]
    f = grid.features_from_lines(*SITE, lines)
    assert f.dist_km == pytest.approx(0.0, abs=0.01) and f.dist_230_km == pytest.approx(7.0, abs=0.05)
    assert f.n_lines_10km == 2.0 and f.max_kv_10km == 345


def test_nothing_nearby_is_capped_at_the_radius_and_far_lines_are_ignored():
    far = line(500, SITE[1] + km_east(60), 39.5, SITE[1] + km_east(60), 40.5)
    f = grid.features_from_lines(*SITE, [far])
    assert (f.dist_km, f.dist_230_km, f.n_lines_10km, f.max_kv_10km) == (grid.RADIUS_KM, grid.RADIUS_KM, 0.0, 0.0)
    assert grid.features_from_lines(*SITE, []).dist_km == grid.RADIUS_KM
    mid = line(230, SITE[1] + km_east(18), 39.5, SITE[1] + km_east(18), 40.5)  # inside 25 km but outside 10 km
    g = grid.features_from_lines(*SITE, [mid])
    assert g.dist_km == pytest.approx(18.0, abs=0.1) and g.n_lines_10km == 0.0


def test_degenerate_lines_are_skipped_not_fatal():
    ok = line(115, SITE[1] + km_east(5), 39.5, SITE[1] + km_east(5), 40.5)
    assert grid.features_from_lines(*SITE, [(115, np.array([[SITE[1], SITE[0]]])), ok]).dist_km == pytest.approx(5.0, abs=0.05)


def test_voltage_falls_back_to_the_voltage_class_when_the_number_is_missing():
    assert grid.voltage_kv(345, "345") == 345 and grid.voltage_kv(-999999, "220-287") == 230.0
    assert grid.voltage_kv(None, "735 AND ABOVE") == 765.0 and grid.voltage_kv(None, "weird") == 115.0


def test_service_response_is_parsed_into_lines_and_failures_become_grid_unavailable(monkeypatch):
    payload = {"features": [{"attributes": {"VOLTAGE": 230, "VOLT_CLASS": "220-287"}, "geometry": {"paths": [[[-83.0, 40.0], [-82.9, 40.0]], [[1, 1]]]}}]}

    class R:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return self.body
    monkeypatch.setattr(grid.urllib.request, "urlopen", lambda *a, **k: R(json.dumps(payload).encode()))
    lines = grid.fetch_lines_near(40.0, -83.0)
    assert len(lines) == 1 and lines[0][0] == 230 and lines[0][1].shape == (2, 2)  # the one-vertex path is dropped
    monkeypatch.setattr(grid.urllib.request, "urlopen", lambda *a, **k: R(b'{"error": {"message": "token required"}}'))
    with pytest.raises(grid.GridUnavailable, match="token required"):
        grid.fetch_lines_near(40.0, -83.0)

    def boom(*a, **k):
        raise OSError("dns")
    monkeypatch.setattr(grid.urllib.request, "urlopen", boom)
    with pytest.raises(grid.GridUnavailable, match="unreachable"):
        grid.load_grid(40.0, -83.0)
