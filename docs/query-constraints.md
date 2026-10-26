# Query constraints

## Coordinate contract

The frontend recognizes coordinate input with the same lightweight module used by the backend (`api/query_contract.py`). It forwards invalid coordinates too, so the backend's authoritative validation can explain the error rather than labeling the request unrelated to ocean science.

Supported examples include `lat 70 long 40`, `latitude 70 longitude 40`, `latitude: 70, longitude: 40`, signed values, and `70 N, 40 E`. Latitude must be in [-90, 90]; longitude in [-180, 180]. South/west indicators produce negative coordinates. Negative values combined with hemisphere indicators are rejected for clarification rather than silently changing signs. The existing point-search radius remains configurable with `FLOATCHAT_SEARCH_RADIUS_DEG` (default 1 degree); a point at 70,40 searches latitude [69,71], longitude [39,41].

## Dates

All calendar interpretation uses **UTC**. `parse_query(text, reference_date=...)` and `query_profiles(text, reference_date=...)` accept an injected date or datetime. Aware datetimes are converted to UTC; naive datetimes are interpreted as UTC.

At reference date **2026-10-26**:

| Expression | Inclusive calendar dates |
| --- | --- |
| last month | 2026-09-01 through 2026-09-30 |
| this month | 2026-10-01 through 2026-10-31 (whole calendar month) |
| last week | 2026-10-19 through 2026-10-25 (previous Monday–Sunday) |

Explicit ISO dates take precedence over relative expressions. Invalid dates and unsupported relative date expressions require clarification. The recent default is applied only when no user date constraint exists. End dates are implemented as an exclusive next-midnight bound, so fractional seconds on the final day are included.

## Depth

`between 0 and 1000 metres depth` means **0 <= depth_m <= 1000**, with both boundaries inclusive. Explicit exclusive bounds require clarification; they are not silently interpreted as inclusive. Invalid, negative, reversed, or over-12000-m bounds are rejected.

The dataset measures pressure in **decibars**, not depth in metres. Depth is derived from measured pressure and the profile latitude using the [UNESCO 1983 standard-ocean pressure/depth algorithm](https://www.jodc.go.jp/info/ioc_doc/UNESCO_tech/059832eb.pdf). This is an estimate, not a separate depth sensor reading; pressure is never relabeled as metres. The implementation matches the published check value: 10000 dbar at latitude 30 degrees gives 9712.653 m.

ERDDAP receives conservative pressure bounds covering the requested depth interval at all latitudes. Returned levels are then filtered using their own profile latitude. The cache uses the same depth equation in SQL before profile and level LIMITs, with an additional check on returned levels. A depth subset does not overwrite a complete cached profile. Valid zero temperatures/salinities are preserved alongside their original pressures; `depth_m` is additive.

The existing 15-level response cap is independent of the requested depth interval. Matching levels are selected before that cap. A bounded response is not a complete vertical profile or exhaustive regional census.

## Empty and incomplete results

The list-shaped API contract and existing fields remain intact. The first response item additionally contains `filters_used` and `retrieval_status`:

- `empty`: successful, complete retrieval found no observations under the applied constraints. The explanation lists those constraints and does not assert absence outside them.
- `failed`: upstream timeout, unavailable dataset/endpoint, API error, or invalid response; never described as no matching observations.
- `incomplete`: an empty bounded/partially parsed sample, unavailable depth information, or cache-only search cannot establish zero matches.
- `success` / `partial`: observations were returned; `partial` marks a cache sample or retrieval that reached a safety cap or discarded invalid records.
- `invalid_filters`: validation/clarification response before retrieval.

A generic HTTP 404 is a retrieval failure. Only ERDDAP's explicit no-matching-results response establishes a successful zero. Structured queries do not fall back to semantic results that could ignore the requested dates, coordinates, or depths.

## Reproducible example

Input, with reference date 2026-10-26:

```text
Find temperature and salinity observations between 0 and 1000 metres depth. lat 70 long 40 last month
```

Parsed constraints:

```json
{
  "latitude": [69.0, 71.0],
  "longitude": [39.0, 41.0],
  "date_start": "2026-09-01",
  "date_end": "2026-09-30",
  "min_depth_m": 0.0,
  "max_depth_m": 1000.0,
  "timezone": "UTC",
  "depth_bounds": "inclusive"
}
```

The corresponding upstream constraints include `time >= 2026-09-01T00:00:00Z`, `time < 2026-10-01T00:00:00Z`, the coordinate box above, and conservative pressure bounds approximately [0,1013.321657] dbar. Final membership uses derived metres at the actual latitude, rather than that broad pressure prefilter alone.

In the **synthetic regression fixture**, pressure levels [0,500,1500] dbar at latitude 70 produce returned pressures [0,500] and derived depths [0,494.3397601042] m. The 1500-dbar level is excluded. Fixture temperatures/salinities [0/0,7/35] remain unchanged. These numbers demonstrate tested filtering; they are not newly retrieved ocean observations.

## Validation and changed files

`python -m pytest -q -ra`: **175 passed, 34 subtests passed**. `python -m compileall -q api frontend ingestion` and `git diff --check` passed.

Tests cover frontend/backend coordinate forms and invalid bounds, signs/hemispheres, UTC and leap/year boundaries, relative/explicit date precedence, end-to-end propagation through Flask and the actual provider implementation with mocked HTTP, inclusive depth filtering, the published UNESCO reference, executed cache SQL in an isolated SQLite database, filtering before profile/level caps, valid zero measurements, and confirmed zero versus failed/incomplete retrieval.

Changed implementation: `api/query_contract.py` (new), `api/query_parser.py`, `api/argo_provider.py`, `api/argo_cache.py`, `api/query.py`, `frontend/chatbot_ui.py`. Tests: `tests/test_query_constraints.py` (new), `tests/test_live_integration.py`. This document is new. No dependencies or deployment settings changed.

These changes are local and have not been deployed as part of this task.
