"""
api/argo_cache.py — PostgreSQL cache for Argo profiles retrieved from remote sources.

Implements idempotent upserts using the unique index on (wmo_id, juld).
Provides cache lookup before remote fetch, freshness checks, and provenance tracking.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger("floatchat.argo_cache")

# ---------------------------------------------------------------------------
# Cache freshness: cached remote profiles older than this are considered stale
# ---------------------------------------------------------------------------
_CACHE_TTL_HOURS = int(os.getenv("ARGO_CACHE_TTL_HOURS", "24"))

# Embedding model — shared instance via centralized model_manager
def _get_embedding_model() -> Any:
    from api.model_manager import get_shared_sentence_transformer
    return get_shared_sentence_transformer("all-MiniLM-L6-v2")


def _make_embedding(wmo_id: str, lat: float, lon: float, juld: datetime) -> list[float]:
    """Generate an embedding for a profile (used for semantic search)."""
    desc = (
        f"Ocean profile WMO {wmo_id} at lat {lat:.3f}, lon {lon:.3f}, "
        f"date {juld.strftime('%Y-%m-%d')}"
    )
    return _get_embedding_model().encode(desc).tolist()


# ---------------------------------------------------------------------------
# Cache operations
# ---------------------------------------------------------------------------

def upsert_profiles(
    conn: "psycopg2.extensions.connection",
    profiles: list[dict],
    skip_embedding: bool = False,
) -> tuple[int, int]:
    """
    Upsert a list of normalised Argo profile dicts into PostgreSQL.

    Uses ON CONFLICT (wmo_id, juld) WHERE wmo_id IS NOT NULL to prevent
    duplicates while allowing updates when source data is refreshed.

    Returns (inserted_count, updated_count).
    """
    inserted = 0
    updated = 0

    with conn.cursor() as cur:
        for profile in profiles:
            wmo_id: str = str(profile.get("wmo_id", "")).strip()
            juld: datetime = profile["juld"]
            lat: float = float(profile["latitude"])
            lon: float = float(profile["longitude"])
            source: str = profile.get("source", "erddap_ifremer")
            source_url: Optional[str] = profile.get("source_url")
            retrieved_at = profile.get("retrieved_at") or datetime.now(tz=timezone.utc).isoformat()
            data_mode: Optional[str] = profile.get("data_mode")
            depth_levels: list[dict] = profile.get("depth_levels", [])

            # Generate embedding
            embedding = profile.get("_embedding")
            if embedding is None and not skip_embedding:
                try:
                    embedding = _make_embedding(wmo_id, lat, lon, juld)
                except Exception as exc:
                    logger.warning("Embedding failed for WMO %s: %s", wmo_id, exc)
                    embedding = [0.0] * 384

            # ---------- Upsert profile row ----------
            cur.execute(
                """
                INSERT INTO profiles
                    (wmo_id, juld, latitude, longitude, source, source_url,
                     retrieved_at, data_mode, embedding, n_prof)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (wmo_id, juld) WHERE wmo_id IS NOT NULL
                DO UPDATE SET
                    latitude     = EXCLUDED.latitude,
                    longitude    = EXCLUDED.longitude,
                    source       = EXCLUDED.source,
                    source_url   = EXCLUDED.source_url,
                    retrieved_at = EXCLUDED.retrieved_at,
                    data_mode    = EXCLUDED.data_mode,
                    embedding    = EXCLUDED.embedding
                RETURNING id,
                    (xmax = 0) AS was_inserted
                """,
                (
                    wmo_id or None,
                    juld,
                    lat,
                    lon,
                    source,
                    source_url,
                    retrieved_at,
                    data_mode,
                    embedding,
                    0,  # n_prof placeholder
                ),
            )
            row = cur.fetchone()
            profile_id, was_inserted = row
            if was_inserted:
                inserted += 1
            else:
                updated += 1

            # ---------- Upsert level rows ----------
            if depth_levels:
                # Delete existing levels only on update (re-import)
                if not was_inserted:
                    cur.execute(
                        "DELETE FROM profile_levels WHERE profile_id = %s",
                        (profile_id,),
                    )

                level_rows = []
                for idx, lvl in enumerate(depth_levels):
                    pres = _safe_float(lvl.get("pres"))
                    temp = _safe_float(lvl.get("temp"))
                    psal = _safe_float(lvl.get("psal") or lvl.get("salinity"))
                    pres_qc = lvl.get("pres_qc")
                    temp_qc = lvl.get("temp_qc")
                    psal_qc = lvl.get("psal_qc")

                    if pres is None and temp is None and psal is None:
                        continue

                    level_rows.append(
                        (profile_id, idx, pres, temp, psal, juld, pres_qc, temp_qc, psal_qc)
                    )

                if level_rows:
                    psycopg2.extras.execute_batch(
                        cur,
                        """
                        INSERT INTO profile_levels
                            (profile_id, n_levels, pres, temp, psal, juld,
                             pres_qc, temp_qc, psal_qc)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        level_rows,
                        page_size=500,
                    )

        conn.commit()

    logger.info("Cache upsert: %d inserted, %d updated", inserted, updated)
    return inserted, updated


def lookup_cached_profiles(
    conn: "psycopg2.extensions.connection",
    wmo_ids: list[str] | None = None,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    source: str | None = None,
    max_results: int = 50,
    require_fresh: bool = False,
) -> list[dict]:
    """
    Query cached profiles with structured filters.

    Returns list of profile dicts with depth_levels populated.
    When require_fresh=True, only returns profiles retrieved within CACHE_TTL_HOURS.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if wmo_ids:
        placeholders = ",".join(["%s"] * len(wmo_ids))
        conditions.append(f"wmo_id IN ({placeholders})")
        params.extend(wmo_ids)

    if min_lat is not None:
        conditions.append("latitude >= %s")
        params.append(min_lat)
    if max_lat is not None:
        conditions.append("latitude <= %s")
        params.append(max_lat)
    if min_lon is not None:
        conditions.append("longitude >= %s")
        params.append(min_lon)
    if max_lon is not None:
        conditions.append("longitude <= %s")
        params.append(max_lon)

    if date_start:
        conditions.append("juld >= %s")
        params.append(date_start)
    if date_end:
        conditions.append("juld <= %s")
        params.append(date_end + " 23:59:59")

    if source:
        conditions.append("source = %s")
        params.append(source)

    if require_fresh:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_CACHE_TTL_HOURS)
        conditions.append("(retrieved_at IS NULL OR retrieved_at >= %s)")
        params.append(cutoff.isoformat())

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    params.append(max_results)

    sql = f"""
        SELECT id, wmo_id, juld, latitude, longitude,
               source, source_url, retrieved_at, data_mode
        FROM profiles
        WHERE {where_clause}
        ORDER BY juld DESC
        LIMIT %s
    """

    results: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

        for row in rows:
            if len(row) >= 9:
                (pid, wmo_id, juld, lat, lon, source_val, source_url, retrieved_at, data_mode) = row[:9]
            elif len(row) >= 5:
                # Fallback / mock handling for partial SELECT tuples
                pid, lat, lon, juld = row[0], row[1], row[2], row[3]
                wmo_id = None
                source_val = "local"
                source_url = None
                retrieved_at = None
                data_mode = None
            else:
                continue

            # Fetch depth levels
            cur.execute(
                """
                SELECT pres, temp, psal, pres_qc, temp_qc, psal_qc
                FROM profile_levels
                WHERE profile_id = %s
                ORDER BY n_levels ASC
                LIMIT 200
                """,
                (pid,),
            )
            levels_rows = cur.fetchall()

            depth_levels = []
            for lvl_row in levels_rows:
                if len(lvl_row) >= 6:
                    pres, temp, psal, pres_qc, temp_qc, psal_qc = lvl_row[:6]
                elif len(lvl_row) >= 3:
                    pres, temp, psal = lvl_row[:3]
                    pres_qc = temp_qc = psal_qc = None
                else:
                    continue
                if pres is None and temp is None and psal is None:
                    continue
                depth_levels.append(
                    {
                        "pres": pres,
                        "temp": temp,
                        "salinity": psal,
                        "psal": psal,
                        "pres_qc": pres_qc,
                        "temp_qc": temp_qc,
                        "psal_qc": psal_qc,
                    }
                )

            juld_str = juld.strftime("%Y-%m-%d %H:%M:%S") if juld else "unknown"

            # Determine cache status for frontend display
            if retrieved_at:
                age_hours = (datetime.now(tz=timezone.utc) - retrieved_at.replace(tzinfo=timezone.utc)).total_seconds() / 3600
                cache_status = "live" if age_hours < 1 else "cached"
            else:
                cache_status = "local"

            results.append(
                {
                    "profile_id": pid,
                    "wmo_id": wmo_id,
                    "lat": lat,
                    "lon": lon,
                    "time": juld_str,
                    "source": source_val or "local",
                    "source_url": source_url,
                    "retrieved_at": retrieved_at.isoformat() if retrieved_at else None,
                    "data_mode": data_mode,
                    "cache_status": cache_status,
                    "depth_levels": depth_levels,
                }
            )

    return results


def is_cache_fresh(
    conn: "psycopg2.extensions.connection",
    wmo_ids: list[str],
) -> bool:
    """Return True if all given WMO IDs have fresh cached data."""
    if not wmo_ids:
        return False
    cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=_CACHE_TTL_HOURS)
    placeholders = ",".join(["%s"] * len(wmo_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT COUNT(*) FROM profiles
            WHERE wmo_id IN ({placeholders})
              AND retrieved_at >= %s
            """,
            [*wmo_ids, cutoff.isoformat()],
        )
        count = cur.fetchone()[0]
    return count >= len(wmo_ids)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None
