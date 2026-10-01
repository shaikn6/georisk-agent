"""The real imagery path, exercised with fake catalog items and a real (tiny) GeoTIFF: no network."""

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from georisk.aoi import BBox
from georisk.providers import _common, _stac
from georisk.providers.base import NoSceneFound, ProviderError
from georisk.providers.earth_search import EarthSearchProvider
from georisk.providers.planetary import PlanetaryComputerProvider
from georisk.scene import BANDS

BBOX = BBox(-83.02, 39.95, -83.0, 39.97)
NEW = datetime(2026, 9, 25, tzinfo=timezone.utc)  # processing baseline >= 04.00: DN carries a +1000 offset
OLD = datetime(2021, 6, 1, tzinfo=timezone.utc)


def item(name="S2A_X", when=NEW, cloud=5.0, assets=None):
    keys = assets or list(EarthSearchProvider.assets.values()) + ["scl"]
    return SimpleNamespace(id=name, datetime=when, properties={"eo:cloud_cover": cloud, "platform": "sentinel-2a"},
                           assets={k: SimpleNamespace(href=f"cog://{k}") for k in keys})


@pytest.fixture
def fake_reads(monkeypatch):
    """Every band reads as DN 3000; the SCL has cloud (class 9) in the top-left quarter."""
    def read(href, bbox, shape, fill_value=0):
        if href.endswith("scl"):
            scl = np.full(shape, 4, dtype="uint8")
            scl[: shape[0] // 2, : shape[1] // 2] = 9
            return scl
        return np.full(shape, 3000.0)
    monkeypatch.setattr(_common, "read_window", read)


def test_build_scene_applies_the_baseline_offset_only_for_new_scenes(fake_reads):
    new = _common.build_scene(item(when=NEW), EarthSearchProvider.assets, "scl", BBOX, 16, "earth-search")
    old = _common.build_scene(item(when=OLD), EarthSearchProvider.assets, "scl", BBOX, 16, "earth-search")
    assert np.allclose(new.bands["red"], 0.2) and np.allclose(old.bands["red"], 0.3)  # 3000*1e-4 (-0.1 after the baseline change)
    assert set(BANDS) <= set(new.bands)
    assert new.source == "earth-search:S2A_X" and new.cloud_cover == 5.0 and new.meta["grid"] == 16


def test_cloud_pixels_are_masked_out(fake_reads):
    sc = _common.build_scene(item(), EarthSearchProvider.assets, "scl", BBOX, 16, "earth-search")
    assert sc.valid.shape == (16, 16) and not sc.valid[:8, :8].any() and sc.valid[8:, 8:].all()


def test_build_scenes_skips_a_broken_item_but_fails_if_all_are_broken(monkeypatch, fake_reads):
    real = _common.read_window

    def flaky(href, bbox, shape, fill_value=0):
        if "S2_BAD" in str(flaky.current):
            raise OSError("cog unreadable")
        return real(href, bbox, shape, fill_value)

    items = [item("S2_BAD"), item("S2_GOOD")]
    orig_build = _common.build_scene

    def build(it, *a, **k):
        flaky.current = it.id
        return orig_build(it, *a, **k)
    monkeypatch.setattr(_common, "build_scene", build)
    monkeypatch.setattr(_common, "read_window", flaky)
    out = _common.build_scenes(items, EarthSearchProvider.assets, "scl", BBOX, 8, "earth-search")
    assert [s.source for s in out] == ["earth-search:S2_GOOD"]
    with pytest.raises(RuntimeError, match="cog unreadable"):
        _common.build_scenes([item("S2_BAD")], EarthSearchProvider.assets, "scl", BBOX, 8, "earth-search")


class FakeClient:
    def __init__(self, items):
        self._items = items

    def search(self, **kw):
        self.kw = kw
        return SimpleNamespace(items=lambda: iter(self._items))


def provider(items, monkeypatch):
    p = EarthSearchProvider()
    monkeypatch.setattr(p, "_client", lambda: FakeClient(items))
    return p


def test_search_prefers_the_clearest_scene_and_reports_empty_results(monkeypatch):
    p = provider([item("B", cloud=30), item("A", cloud=2), item("C", cloud=12)], monkeypatch)
    assert [i.id for i in p._search(BBOX, 60, 40.0)] == ["A", "C", "B"]
    with pytest.raises(NoSceneFound):
        provider([], monkeypatch)._search(BBOX, 60, 40.0)


def test_load_builds_the_clearest_scene_and_maps_failures_to_provider_errors(monkeypatch, fake_reads):
    sc = provider([item("B", cloud=30), item("A", cloud=2)], monkeypatch).load(BBOX, days=60, max_cloud=40, grid=8)
    assert sc.source == "earth-search:A"
    with pytest.raises(NoSceneFound):  # "no scene" passes through untouched
        provider([], monkeypatch).load(BBOX, days=60, max_cloud=40, grid=8)
    p = provider([item()], monkeypatch)
    monkeypatch.setattr(_stac, "build_scene", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad asset")))
    with pytest.raises(ProviderError, match="bad asset"):
        p.load(BBOX, days=60, max_cloud=40, grid=8)


def test_load_series_takes_one_scene_per_date_up_to_n(monkeypatch, fake_reads):
    same_day = datetime(2026, 9, 25, 10, tzinfo=timezone.utc)
    items = [item("A", NEW, 1), item("A2", same_day, 3), item("B", datetime(2026, 9, 20, tzinfo=timezone.utc), 4),
             item("C", datetime(2026, 9, 15, tzinfo=timezone.utc), 5)]
    out = provider(items, monkeypatch).load_series(BBOX, days=60, max_cloud=40, grid=8, n=2)
    assert [s.source.split(":")[1] for s in out] == ["A", "B"]  # A2 is the same date as A; capped at n=2
    with pytest.raises(NoSceneFound):
        provider([], monkeypatch).load_series(BBOX, days=60, max_cloud=40, grid=8, n=2)


def test_probe_maps_connection_errors(monkeypatch):
    def boom(*a, **k):
        raise OSError("dns")
    monkeypatch.setattr(_stac.Client, "open", boom)
    with pytest.raises(ProviderError, match="unreachable"):
        EarthSearchProvider().probe()
    monkeypatch.setattr(_stac.Client, "open", lambda *a, **k: object())
    EarthSearchProvider().probe()  # reachable: no error


def test_planetary_computer_signs_its_requests(monkeypatch):
    seen = {}
    monkeypatch.setattr("georisk.providers.planetary.Client.open", lambda url, **kw: seen.update(url=url, **kw) or "client")
    assert PlanetaryComputerProvider()._client() == "client" and callable(seen["modifier"]) and "planetarycomputer" in seen["url"]
    assert set(BANDS) <= set(PlanetaryComputerProvider.assets)


def test_read_window_reads_the_right_pixels_and_fills_outside_the_raster(tmp_path):
    """A real GeoTIFF in a UTM projection, read by lon/lat bbox: exercises the reprojection and window maths."""
    path = tmp_path / "t.tif"
    data = np.arange(100 * 100, dtype="float32").reshape(100, 100)
    # Columbus, OH is in UTM zone 17N (EPSG:32617); origin chosen so the raster covers a few km around the AOI
    from pyproj import Transformer

    x0, y0 = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True).transform(-83.02, 39.97)
    with rasterio.open(path, "w", driver="GTiff", height=100, width=100, count=1, dtype="float32", crs="EPSG:32617",
                       transform=from_origin(x0 - 500, y0 + 500, 30, 30), nodata=-1) as ds:
        ds.write(data, 1)
    inside = _common.read_window(str(path), BBOX, (20, 20))
    assert inside.shape == (20, 20) and np.isfinite(inside).all() and inside.max() > inside.min()  # a real gradient came back
    far = BBox(-80.0, 30.0, -79.99, 30.01)  # nowhere near the raster
    assert np.all(_common.read_window(str(path), far, (4, 4), fill_value=-7) == -7)  # boundless read fills, never invents
