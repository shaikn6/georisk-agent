"""Microsoft Planetary Computer: an independent second source (token signing is keyless)."""

from __future__ import annotations

from typing import ClassVar

import planetary_computer
from pystac_client import Client

from ._stac import StacProvider


class PlanetaryComputerProvider(StacProvider):
    name = "planetary-computer"
    url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    assets: ClassVar[dict[str, str]] = {"green": "B03", "red": "B04", "nir": "B08", "swir1": "B11", "swir2": "B12"}
    scl_asset = "SCL"

    def _client(self) -> Client:
        return Client.open(self.url, modifier=planetary_computer.sign_inplace, timeout=12)
