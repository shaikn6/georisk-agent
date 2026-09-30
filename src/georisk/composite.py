"""Multi-date compositing: a per-pixel median over several clear scenes.

Reduces cloud leakage, one-off haze and single-date seasonality that a lone scene cannot.
"""

from __future__ import annotations

import numpy as np

from .scene import BANDS, Scene


def median_composite(scenes: list[Scene]) -> Scene:
    """Median of each band over pixels that are valid *in that scene*.

    A pixel is valid in the composite if at least one scene saw it clearly.
    """
    if not scenes:
        raise ValueError("need at least one scene")
    if len(scenes) == 1:
        s = scenes[0]
        return Scene(s.bands, s.valid, s.source, s.acquired, s.cloud_cover, {**s.meta, "n_scenes": 1})
    shapes = {s.valid.shape for s in scenes}
    if len(shapes) != 1:
        raise ValueError(f"scenes are not co-registered: {shapes}")
    valid = np.stack([s.valid for s in scenes])
    bands = {}
    for b in BANDS:
        stack = np.stack([s.bands[b] for s in scenes]).astype("float64")
        stack[~valid] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(stack, axis=0)
        bands[b] = np.nan_to_num(med, nan=0.0)
    dates = sorted(s.acquired for s in scenes if s.acquired)
    clouds = [s.cloud_cover for s in scenes if s.cloud_cover is not None]
    return Scene(
        bands=bands,
        valid=valid.any(axis=0),
        source=f"composite[{len(scenes)}]:" + ",".join(s.source.split(":")[-1] for s in scenes),
        acquired=dates[-1] if dates else None,
        cloud_cover=float(np.mean(clouds)) if clouds else None,
        meta={"n_scenes": len(scenes), "first_date": dates[0] if dates else None,
              "provider": scenes[0].meta.get("provider")},
    )
