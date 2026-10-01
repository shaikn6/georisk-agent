FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 GEORISK_CACHE_DIR=/data/cache
WORKDIR /app
# rasterio's bundled GDAL needs libexpat, which the slim base image lacks (live imagery fails with ImportError without it)
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd -m app && mkdir -p /data/cache && chown -R app /data
USER app
EXPOSE 8000
# Liveness never touches external services, so the container is not restarted for an upstream outage.
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)"
CMD ["uvicorn", "georisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
