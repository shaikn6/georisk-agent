import pytest

from georisk.aoi import bbox_from_point


def test_box_area_matches_radius():
    b = bbox_from_point(40.0, -83.0, 1000)  # 2 km x 2 km
    assert b.area_km2() == pytest.approx(4.0, rel=0.02)
    assert b.west < -83.0 < b.east and b.south < 40.0 < b.north


def test_high_latitude_box_is_still_square_in_metres():
    assert bbox_from_point(64.0, 10.0, 500).area_km2() == pytest.approx(1.0, rel=0.02)


@pytest.mark.parametrize("lat,lon,r", [(91, 0, 100), (0, 181, 100), (0, 0, 5), (0, 0, 50_000)])
def test_rejects_bad_input(lat, lon, r):
    with pytest.raises(ValueError):
        bbox_from_point(lat, lon, r)
