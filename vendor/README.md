`libexpat.so.1` is the unmodified Ubuntu 20.04 `libexpat1` 2.2.9 build (MIT license, https://libexpat.github.io/).
Vercel's Python runtime lacks it, and rasterio's GDAL needs it; `app.py` preloads it on Linux before importing rasterio.
