"""Vercel entrypoint: exposes the FastAPI app from the src/ layout without needing an editable install."""

import ctypes
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("GEORISK_CACHE_DIR", "/tmp/georisk_cache")  # the deployment filesystem is read-only except /tmp
os.environ.setdefault("GEORISK_MAX_INFLIGHT", "4")
os.environ.setdefault("GEORISK_TRUST_PROXY", "1")  # Vercel's edge sets X-Forwarded-For; it is not client-controlled there

if sys.platform.startswith("linux"):
    # The serverless runtime lacks libexpat, which rasterio's GDAL needs; preload the bundled copy.
    _lib = os.path.join(os.path.dirname(__file__), "vendor", "libexpat.so.1")
    if os.path.exists(_lib):
        ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

from georisk.api import app  # noqa: F401
