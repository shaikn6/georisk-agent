import json
from types import SimpleNamespace

import numpy as np
import pytest

from georisk import terrain
from georisk.aoi import BBox
from georisk.providers import _common


def dem_item():
    return SimpleNamespace(assets={"data": SimpleNamespace(href="cog://dem")})


@pytest.fixture
def stac(monkeypatch):
    """Fake DEM catalog; tests set `.items` and `.read` to control what comes back."""
    state = SimpleNamespace(items=[dem_item()], read=None)
    monkeypatch.setattr("pystac_client.Client.open", lambda *a, **k: SimpleNamespace(
        search=lambda **kw: SimpleNamespace(items=lambda: iter(state.items))))
    monkeypatch.setattr(_common, "read_window", lambda href, bbox, shape, fill_value=0: state.read(shape))
    return state


def valley(shape):
    r, c = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (1.5 * np.abs(c - shape[1] // 2) + 0.02 * (shape[0] - 1 - r)).astype("float32")


def test_load_terrain_returns_slope_and_relief_from_the_dem(stac):
    stac.read = valley
    t = terrain.load_terrain(BBox(-83.02, 39.95, -83.0, 39.97), grid=32)
    assert t.relief_m > 10 and t.mean_slope_deg > 0 and 0 <= t.center_percentile <= 1


def test_nodata_sentinel_is_not_treated_as_elevation(stac):
    def read(shape):
        d = valley(shape)
        d[:2, :2] = -32767  # DEM nodata
        return d
    stac.read = read
    assert terrain.load_terrain(BBox(-83.02, 39.95, -83.0, 39.97), grid=32).relief_m < 150  # a -32767 m "pit" would give ~33 km


def test_missing_tiles_and_read_failures_become_terrain_unavailable(stac):
    stac.items = []
    with pytest.raises(terrain.TerrainUnavailable, match="no DEM tile"):
        terrain.load_terrain(BBox(-83.02, 39.95, -83.0, 39.97))
    stac.items = [dem_item()]
    stac.read = lambda shape: (_ for _ in ()).throw(OSError("host down"))
    with pytest.raises(terrain.TerrainUnavailable, match="host down"):
        terrain.load_terrain(BBox(-83.02, 39.95, -83.0, 39.97))


def test_load_hydrology_finds_the_stream_in_a_synthetic_valley(stac):
    stac.read = lambda shape: valley(shape).astype("float64")
    h = terrain.load_hydrology(39.96, -83.01)
    assert h.dist_to_stream_m < 150 and h.hand_mean_m >= 0 and h.near_stream_fraction > 0


def test_load_hydrology_refuses_a_mostly_missing_dem_and_survives_small_gaps(stac):
    def holes(frac):
        def read(shape):
            d = valley(shape).astype("float64")
            d[: int(shape[0] * frac), :] = np.nan
            return d
        return read
    stac.read = holes(0.5)
    with pytest.raises(terrain.TerrainUnavailable, match="missing"):
        terrain.load_hydrology(39.96, -83.01)
    stac.read = holes(0.02)  # 2% gaps are filled, not fatal
    assert terrain.load_hydrology(39.96, -83.01).hand_mean_m >= 0


def test_cli_prints_a_full_assessment_for_an_offline_scene(monkeypatch, capsys):
    from georisk import cli

    monkeypatch.setattr("sys.argv", ["georisk", "--lat", "39.96", "--lon", "-83.0", "--offline", "floodplain", "--no-llm"])
    cli.main()
    out = json.loads(capsys.readouterr().out)
    assert out["overall"] is not None and out["narrative"] and {h["name"] for h in out["hazards"]} >= {"flood", "wildfire"}


def test_cli_rejects_an_unknown_offline_scene(monkeypatch):
    from georisk import cli

    monkeypatch.setattr("sys.argv", ["georisk", "--lat", "1", "--lon", "1", "--offline", "atlantis"])
    with pytest.raises(SystemExit):
        cli.main()
