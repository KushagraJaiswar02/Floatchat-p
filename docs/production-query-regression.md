# Production query timeout investigation

Investigated 2026-10-26 using the frontend's exact request shape:

```http
POST /query
Content-Type: application/json

{"query":"Find Argo float observations of temperature and salinity in the Indian Ocean."}
```

## Evidence before the fix

- The production request returned HTTP 200 after 32.859 seconds, beyond the frontend's 30-second read timeout.
- Local Flask execution with real database/upstream access took 35.658 seconds: database connect 1.532 s, migrations 2.977 s, cache lookup 0.230 s, **ERDDAP 30.650 s (timeout)**, semantic fallback 0.227 s.
- Render runtime logs independently showed ERDDAP starting at 16:40:15.677 UTC and timing out at 16:40:46.315 UTC.
- The parsed intent was correctly `profile_search`. The all-history Indian Ocean bounding box reached ERDDAP with `orderByLimit("6000")`, without a date constraint. The row cap bounded output, not the cost of searching the archive.
- Render events showed runtime OOM kills at 15:48:31 and 15:52:43 UTC, explicitly against a `512Mi` limit. A later worker used 468,799,500 bytes before this fix. Runtime logs also showed Gemini quota errors; no successful AI explanation is assumed.
- The memory refactor shared the model instance and bounded semantic candidates but left `run_server.py` importing PyTorch eagerly. Cache misses also synchronously initialized the embedding model during profile upserts.

## Changes

- Added payload-free start/end stage logs, request IDs, elapsed milliseconds, error types, process IDs, and Linux process peak RSS. Covered classification, DB connect/migration/cache operations, ERDDAP HTTP/parse, embedding/model initialization, semantic search, Gemini initialization/calls, formatting, serialization, and total request execution.
- Undated spatial requests without a float ID search yesterday and today in UTC and return up to `TOP_K` sampled profiles. Every response discloses this window and asks for explicit dates to search another period. Explicit user dates and float IDs retain their requested scope. This is a bounded sample, not an exhaustive regional census or complete depth profile.
- Kept ERDDAP real-data retrieval and provenance. Default HTTP limits are 3 s connect and 10 s read, with configured read timeouts capped at 12 s. Responses stream with a 2 MiB decoded limit and elapsed download checks. Single-read chunks prevent a slow trickle from holding a large buffered chunk indefinitely. Responses close on all paths. No retries, redirects, pagination, or sequential multi-float retry fan-out.
- A failed upstream request returns the existing list-shaped retrieval-error response without loading a semantic model. An empty recent sample reports the actual window rather than searching unrelated undated embeddings.
- Removed startup's eager PyTorch import. Lazy model initialization still applies CPU/single-thread restrictions and the shared singleton remains available for semantic search.
- Structured cache writes skip embeddings, preserve existing embeddings on conflict, and use empty arrays for not-yet-embedded profiles (compatible with the NOT NULL schema). Semantic search excludes empty arrays. No artificial zero vectors are generated.
- Serve normalized real provider observations directly; rollback and log a failed cache write without losing those observations.
- DB connections have a 3 s connect limit; statements/locks have 3 s/1 s limits. Limits use SQL SET because the actual Neon pooled endpoint rejects startup `options` for statement_timeout.
- Gemini has a 4 s timeout and no retries. Existing data-grounded fallback remains available when AI explanation fails.

The Streamlit frontend and its 30-second timeout are unchanged. API responses retain the existing list, profile fields, `query_explain`, provenance, and error conventions; `X-Request-ID` is additive.

## Local verification

- Successful cold-cache reproduction: **17.882 seconds**, HTTP 200, real Ifremer WMO **1902052**, observation time **2026-10-26T03:59:09Z**, location **-15.130, 92.143**, nonempty temperature/salinity levels. ERDDAP 7.518 s including parsing, cache write 2.254 s, explanation/fallback 2.444 s. PyTorch remained unloaded.
- An upstream timeout returned the compatible error response in 15.979 seconds with PyTorch unloaded.
- `python -m pytest -q -ra`: **159 passed, 13 subtests passed**.
- `python -m compileall -q api frontend ingestion` and `git diff --check` passed.
- Regression coverage includes successful structured retrieval without embeddings, preserved explicit dates, upstream connect/read errors, interrupted/oversized/expired downloads, response closure, no multi-float retry fan-out, cache-write failure, NOT NULL embedding handling, Gemini timeout options, and payload-free timing logs. Existing intent and memory tests remain enabled.

Production rollout and memory verification must use the resulting Render deploy, its request IDs, process peak RSS, container memory metrics, and events; local Windows measurements do not establish Render memory usage.

Reference: [ERDDAP tabledap orderByLimit documentation](https://coastwatch.pfeg.noaa.gov/erddap/tabledap/documentation.html).
