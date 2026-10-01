# Changelog

All notable changes. Numbers are leave-region-out (held-out places), never in-sample.

## 0.4.0 — 2026-10-01

### Accuracy and honesty
- **Flood AUC 0.844 → 0.866** (Ohio holdout 0.821 → 0.856): river-height features from the 30 m DEM (`hydrology.py`: fill pits, D8 flow, accumulation, streams, HAND), plus capped inputs for the linear model so it cannot extrapolate on steep sites.
- **Probabilities audited and corrected**: held-out predictions were compressed toward 50%. Platt recalibration ships only when it improves both calibration error and Brier (flood 0.099 → 0.035; fire did not qualify).
- **Track record with every probability**: what happened to similar held-out sites, with a Wilson 95% interval, shown in the API (`evidence`), demo and memo.
- **Fire** gains topography (elevation, ridge/valley position, roughness, aspect); calibration error 0.124 → 0.065.
- **Serving rule**: best post-recalibration Brier among candidates within 0.01 AUC of the best.
- Negative results documented, not hidden: a satellite foundation model (Prithvi) did not help; "unusual site" warnings did not predict error.

### New capabilities
- **Area mode** (`POST /assess/area`, demo Area button): outline in, independent tiles out, GeoJSON/CSV export.
- **Analyst memo**: plain-English risk note built only from the assessment's numbers; optional LLM rewrite is accepted only if every number it writes is in the facts.
- **Experimental solar-siting energy lens**: terrain plus distance to the high-voltage grid vs. the USGS solar-plant database (AUC 0.682; near chance in flat farmland such as Ohio, stated in the artifact, API, memo and demo).
- Demo: place search, shareable `#lat,lon` links, drivers panel, third dial.

### Robustness and operations
- Assessments no longer fail when no clear scene exists: results that need no imagery are returned with a warning (503 only if nothing is computable).
- Single-flight result cache (repeat requests ~2 ms), per-client rate limit, bounded in-flight work, structured JSON logs with request IDs, `/metrics`, `scripts/loadtest.py` (0 5xx at 32 threads).
- Coverage 71% → 83%, enforced at 80% in CI; pip cache and a job timeout in CI; dependency scan clean (only build-tool `pip` flagged).
- Security: API key option, no internal errors in responses, CSP limited to needed hosts, CSV export neutralised against spreadsheet injection, demo page revalidated on deploy.

## 0.3.0 — 2026-09-30
- Calibrated flood and fire probabilities from FEMA NFHL and USGS WFIGS labels, Ohio case study, hardened API, live demo on Vercel.

## 0.2.0 — 2026-09-29
- Resilient architecture: provider failover with circuit breakers, terrain, isolated hazards, confidence score.

## 0.1.0
- Sentinel-2 flood / wildfire-fuel / urban screening pipeline with explainable heuristic scores.
