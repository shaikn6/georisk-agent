import numpy as np
import pytest

from georisk import hydrology as hy


def valley(n=100, wall=1.5, fall=0.02):
    """V-shaped valley: walls rise `wall` m per cell away from column n//2; the floor falls south by `fall` m per row."""
    r, c = np.mgrid[0:n, 0:n]
    return wall * np.abs(c - n // 2) + fall * (n - 1 - r)


def test_height_above_stream_matches_the_known_wall_height():
    dem = valley()
    z = hy.fill_depressions(dem)
    down = hy._downstream(hy.flow_directions(z))
    acc = hy.flow_accumulation(z, down)
    hand, _ = hy.hand_surface(z, down, acc, threshold_cells=100)  # sheet flow on the walls peaks near 50 cells
    assert hand[60, 50] == pytest.approx(0.0, abs=0.05)  # on the stream
    assert hand[60, 60] == pytest.approx(15.0, abs=1.5)  # 10 cells up the wall at 1.5 m/cell
    assert hand[60, 30] == pytest.approx(30.0, abs=2.0)  # 20 cells up the other wall


def test_pits_are_filled_and_all_water_reaches_an_outlet():
    plane = np.add.outer(np.arange(40)[::-1] * 1.0, np.zeros(40))  # slopes south
    plane[20, 20] -= 8  # a pit
    z = hy.fill_depressions(plane)
    assert z[20, 20] >= plane[20, 20] and z.min() >= plane.min() - 1e-9
    down = hy._downstream(hy.flow_directions(z))
    acc = hy.flow_accumulation(z, down)
    outlets = down < 0
    assert acc.ravel()[outlets].sum() == pytest.approx(z.size)  # every cell drains to some outlet: nothing is stranded


def test_flow_accumulation_counts_cells_exactly_on_a_tilted_plane():
    z = np.add.outer(np.arange(30)[::-1] * 1.0, np.zeros(30)) + np.linspace(0, 1e-3, 30)  # slopes south, tiny tilt east-west
    zf = hy.fill_depressions(z)
    acc = hy.flow_accumulation(zf, hy._downstream(hy.flow_directions(zf)))
    assert acc.max() == 900  # one outlet receives all 900 cells (the side tilt routes the bottom row sideways)
    assert acc[0].max() == 1


def test_features_distinguish_a_site_on_the_stream_from_one_on_the_rim():
    dem = valley(120)
    aoi_stream = (slice(50, 70), slice(50, 70))
    aoi_rim = (slice(50, 70), slice(100, 118))
    on = hy.hydrology_features(dem, 30.0, aoi_stream)
    rim = hy.hydrology_features(dem, 30.0, aoi_rim)
    assert on.dist_to_stream_m < rim.dist_to_stream_m and on.hand_mean_m < rim.hand_mean_m / 3
    assert on.near_stream_fraction > 0.02 and rim.near_stream_fraction == 0.0
    assert on.log_stream_area > 0  # the stream it drains to has upstream area


def test_bad_input_is_rejected():
    with pytest.raises(ValueError):
        hy.hydrology_features(np.full((5, 5), 1.0), 30.0, (slice(1, 3), slice(1, 3)))
    bad = valley(40)
    bad[3, 3] = np.nan
    with pytest.raises(ValueError):
        hy.hydrology_features(bad, 30.0, (slice(10, 20), slice(10, 20)))


def test_runtime_is_small_enough_for_a_request():
    import time

    t = time.time()
    hy.hydrology_features(valley(200), 30.0, (slice(80, 120), slice(80, 120)))
    assert time.time() - t < 3.0  # a 6 km window at 30 m must not dominate a ~3 s assessment


def plane(n=60, drop_per_row=0.0, drop_per_col=0.0):
    r, c = np.mgrid[0:n, 0:n]
    return 500.0 + drop_per_row * r + drop_per_col * c  # row index grows southward in a north-up raster


AOI = (slice(20, 40), slice(20, 40))


def test_south_facing_slopes_read_plus_one_and_north_facing_minus_one():
    south = hy.topo_features(plane(drop_per_row=-2.0), 30.0, AOI)  # elevation falls going south: the slope faces south
    north = hy.topo_features(plane(drop_per_row=2.0), 30.0, AOI)
    east = hy.topo_features(plane(drop_per_col=-2.0), 30.0, AOI)  # faces east: no north/south component
    assert south.south_exposure == pytest.approx(1.0, abs=1e-6) and north.south_exposure == pytest.approx(-1.0, abs=1e-6)
    assert abs(east.south_exposure) < 1e-6


def test_ridges_are_upland_and_valleys_are_low_relative_to_the_window():
    n = 60
    _, c = np.mgrid[0:n, 0:n]
    ridge = 100.0 - 1.0 * np.abs(c - 30)  # peak along the centre column
    valley_dem = 100.0 + 1.0 * np.abs(c - 30)
    assert hy.topo_features(ridge, 30.0, AOI).tpi_m > 3 > -3 > hy.topo_features(valley_dem, 30.0, AOI).tpi_m
    assert hy.topo_features(np.full((60, 60), 250.0), 30.0, AOI).south_exposure == 0.0  # flat: no division blow-up


def test_topo_features_report_elevation_and_roughness_and_reject_bad_dems():
    t = hy.topo_features(plane(drop_per_row=-2.0), 30.0, AOI)
    assert t.elev_mean_m == pytest.approx(441.0, abs=0.5) and t.roughness_m > 0  # 500 - 2 m/row * mean row 29.5
    with pytest.raises(ValueError):
        hy.topo_features(np.full((4, 4), 1.0), 30.0, (slice(1, 3), slice(1, 3)))
    bad = np.full((20, 20), 5.0)
    bad[3, 3] = np.inf
    with pytest.raises(ValueError):
        hy.topo_features(bad, 30.0, (slice(5, 15), slice(5, 15)))


def test_flat_fraction_is_one_on_a_plain_and_zero_on_a_steep_face():
    assert hy.topo_features(plane(), 30.0, AOI).flat_fraction == 1.0
    assert hy.topo_features(plane(drop_per_row=-30.0), 30.0, AOI).flat_fraction == 0.0  # 45 degree slope
    gentle = hy.topo_features(plane(drop_per_row=-1.0), 30.0, AOI).flat_fraction  # ~1.9 degrees
    assert gentle == 1.0
