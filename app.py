"""Vercel entrypoint: exposes the FastAPI app from the src/ layout without needing an editable install."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
os.environ.setdefault("GEORISK_CACHE_DIR", "/tmp/georisk_cache")  # the deployment filesystem is read-only except /tmp
os.environ.setdefault("GEORISK_MAX_INFLIGHT", "4")
os.environ.setdefault("GEORISK_TRUST_PROXY", "1")  # Vercel's edge sets X-Forwarded-For; it is not client-controlled there

from georisk.api import app  # noqa: F401
