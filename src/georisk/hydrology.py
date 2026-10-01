"""Hydrology from a DEM: fill pits, D8 flow, flow accumulation, streams, and Height Above Nearest Drainage (HAND).

Flood-zone status is a river phenomenon: what matters is how far a site sits above the nearest stream and how big that
stream is, which a 1 km terrain summary cannot see. Pure numpy/heapq, no GIS stack.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_DIST = np.array([2**0.5, 1, 2**0.5, 1, 1, 2**0.5, 1, 2**0.5])


def fill_depressions(dem: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Priority-flood: raise every pit to its spill point (plus a tiny slope) so water can always leave the grid."""
    h, w = dem.shape
    filled = np.full_like(dem, np.inf, dtype="float64")
    seen = np.zeros(dem.shape, bool)
    heap: list[tuple[float, int, int]] = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                filled[r, c] = dem[r, c]
                seen[r, c] = True
                heapq.heappush(heap, (dem[r, c], r, c))
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc]:
                seen[rr, cc] = True
                filled[rr, cc] = max(dem[rr, cc], z + eps)
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def flow_directions(z: np.ndarray) -> np.ndarray:
    """D8: index (0-7 into _NB) of the steepest downhill neighbour, or -1 for an outlet (grid edge / no lower neighbour)."""
    h, w = z.shape
    pad = np.pad(z, 1, mode="constant", constant_values=np.inf)
    best = np.full((h, w), -1, int)
    steep = np.zeros((h, w))
    for k, (dr, dc) in enumerate(_NB):
        nb = pad[1 + dr : 1 + dr + h, 1 + dc : 1 + dc + w]
        slope = (z - nb) / _DIST[k]
        better = slope > steep
        steep = np.where(better, slope, steep)
        best = np.where(better, k, best)
    return best


def _downstream(fdir: np.ndarray) -> np.ndarray:
    """Flat index of each cell's downstream cell, or -1."""
    h, w = fdir.shape
    idx = np.arange(h * w).reshape(h, w)
    out = np.full((h, w), -1, int)
    for k, (dr, dc) in enumerate(_NB):
        m = fdir == k
        rr, cc = np.nonzero(m)
        out[rr, cc] = idx[rr + dr, cc + dc]
    return out.ravel()


def flow_accumulation(z: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Number of cells draining through each cell (itself included), by visiting cells from high to low."""
    acc = np.ones(z.size)
    for i in np.argsort(-z.ravel(), kind="stable"):
        d = down[i]
        if d >= 0:
            acc[d] += acc[i]
    return acc.reshape(z.shape)


def hand_surface(z_fill: np.ndarray, down: np.ndarray, acc: np.ndarray, threshold_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """(HAND in metres, flat index of the stream cell each cell drains to). Streams are cells with accumulation >= threshold."""
    flat_z, flat_acc = z_fill.ravel(), acc.ravel()
    target = np.full(flat_z.size, -1, int)
    for i in np.argsort(flat_z, kind="stable"):  # downstream cells (lower) are resolved first
        if flat_acc[i] >= threshold_cells:
            target[i] = i
        elif down[i] >= 0:
            target[i] = target[down[i]]
    hand = np.where(target >= 0, flat_z - flat_z[np.maximum(target, 0)], np.nan)
    return hand.reshape(z_fill.shape), target.reshape(z_fill.shape)


@dataclass(frozen=True)
class HydroFeatures:
    hand_mean_m: float          # mean height above the nearest stream over the AOI
    hand_min_m: float
    near_stream_fraction: float  # share of the AOI within 3 m (vertically) of a stream
    dist_to_stream_m: float     # horizontal distance from the AOI centre to the nearest stream cell
    log_stream_area: float      # log10 of the upstream area (km2) of the stream the centre drains to
    hand_big_mean_m: float      # same as hand_mean_m but against only the larger streams (>= 1 km2 upstream)


def hydrology_features(dem: np.ndarray, cell_m: float, aoi_slice: tuple[slice, slice], small_km2: float = 0.09,
                       big_km2: float = 0.9) -> HydroFeatures:
    """Compute AOI-level hydrology from a large DEM window (the AOI is the central region ``aoi_slice``)."""
    dem = np.asarray(dem, dtype="float64")
    if dem.ndim != 2 or min(dem.shape) < 8 or not np.isfinite(dem).all():
        raise ValueError("DEM must be a finite 2-D array of at least 8x8")
    z = fill_depressions(dem)
    down = _downstream(flow_directions(z))
    acc = flow_accumulation(z, down)
    cell_km2 = (cell_m / 1000.0) ** 2
    small, big = max(2, round(small_km2 / cell_km2)), max(3, round(big_km2 / cell_km2))
    hand_s, tgt_s = hand_surface(z, down, acc, small)
    hand_b, _ = hand_surface(z, down, acc, big)
    sub = (hand_s[aoi_slice], hand_b[aoi_slice])
    cy, cx = (aoi_slice[0].start + aoi_slice[0].stop) // 2, (aoi_slice[1].start + aoi_slice[1].stop) // 2
    streams = np.argwhere(acc >= small)
    d = np.hypot(streams[:, 0] - cy, streams[:, 1] - cx).min() * cell_m if len(streams) else float("inf")
    t = tgt_s[cy, cx]
    area = float(acc.ravel()[t] * cell_km2) if t >= 0 else cell_km2
    hs = np.where(np.isfinite(sub[0]), sub[0], np.nanmax(hand_s) if np.isfinite(hand_s).any() else 0.0)
    hb = np.where(np.isfinite(sub[1]), sub[1], np.nanmax(hand_b) if np.isfinite(hand_b).any() else 0.0)
    return HydroFeatures(float(hs.mean()), float(hs.min()), float((hs <= 3.0).mean()), float(min(d, 6000.0)),
                         float(np.log10(max(area, cell_km2))), float(hb.mean()))
