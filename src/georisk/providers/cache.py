"""On-disk scene cache: survives provider outages and avoids re-downloading identical AOIs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np

from ..aoi import BBox
from ..scene import BANDS, Scene

log = logging.getLogger(__name__)

_KEY = re.compile(r"[A-Za-z0-9_-]{1,64}")


class SceneCache:
    def __init__(self, directory: Path, ttl_s: float = 6 * 3600, clock=time.time) -> None:
        self.dir = Path(directory)
        self.ttl_s = ttl_s
        self.clock = clock

    def key(self, bbox: BBox, days: int, max_cloud: float, grid: int) -> str:
        raw = json.dumps([[round(v, 4) for v in bbox.as_list()], days, max_cloud, grid])
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def _path(self, key: str, suffix: str = ".npz") -> Path:
        """Cache file for ``key``; a key that could leave the cache directory is rejected."""
        if not _KEY.fullmatch(key):
            raise ValueError(f"invalid cache key: {key!r}")
        base = os.path.abspath(self.dir)
        full = os.path.normpath(os.path.join(base, f"{key}{suffix}"))
        if not full.startswith(base + os.sep):
            raise ValueError(f"invalid cache key: {key!r}")
        return Path(full)

    def get(self, key: str, *, allow_stale: bool = False) -> Scene | None:
        path = self._path(key)
        if not path.exists():
            return None
        age = self.clock() - path.stat().st_mtime
        if age > self.ttl_s and not allow_stale:
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["meta"]))
                scene = Scene(
                    bands={b: z[b] for b in BANDS},
                    valid=z["valid"],
                    source=meta["source"],
                    acquired=meta.get("acquired"),
                    cloud_cover=meta.get("cloud_cover"),
                    meta={**meta.get("meta", {}), "cache": "stale" if age > self.ttl_s else "hit"},
                )
            return scene
        except Exception as exc:  # noqa: BLE001 - a corrupt cache entry must never break a request
            log.warning("discarding unreadable cache entry %s: %s", path.name, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, scene: Scene) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            # unique per writer
            tmp = self._path(key, f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
            meta = json.dumps(
                {"source": scene.source, "acquired": scene.acquired, "cloud_cover": scene.cloud_cover,
                 "meta": {k: v for k, v in scene.meta.items() if isinstance(v, (str, int, float, type(None)))}}
            )
            np.savez_compressed(tmp, valid=scene.valid, meta=meta, **scene.bands)
            tmp.replace(self._path(key))  # atomic: readers never see a partial file
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            log.warning("cache write failed: %s", exc)
