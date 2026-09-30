"""Element84 Earth Search (AWS open data). No API key."""

from __future__ import annotations

from typing import ClassVar

from ._stac import StacProvider


class EarthSearchProvider(StacProvider):
    name = "earth-search"
    url = "https://earth-search.aws.element84.com/v1"
    assets: ClassVar[dict[str, str]] = {"green": "green", "red": "red", "nir": "nir", "swir1": "swir16", "swir2": "swir22"}
    scl_asset = "scl"
