"""
api/query.py — FloatChat Flask API with live Argo data integration.

Data flow for each /query request:
  1. Parse natural language → ArgoFilters + intent (deterministic, no LLM)
  2. If structured Argo filters found:
       a. Check PostgreSQL cache (fresh data → return immediately)
       b. If cache miss / stale → fetch from ERDDAP
       c. Upsert to cache, return results
  3. Semantic fallback: embedding similarity over cached local profiles
  4. LLM (Gemini) for explanation/interpretation over retrieved data

No raw stack traces to clients. No hardcoded credentials. CORS at module level.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
import psycopg2
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from geopy.geocoders import Nominatim
from geopy.distance import distance

# Local modules
from api.query_parser import parse_query, ParsedQuery
from api.argo_provider import ArgoERDDAPProvider, ArgoFilters, ArgoProviderError
from api.argo_cache import upsert_profiles, lookup_cached_profiles, is_cache_fresh
from api.db_migration import run_migrations

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("floatchat.api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

_DATABASE_URL = os.getenv("DATABASE_URL")
if _DATABASE_URL:
    # Use DATABASE_URL directly (e.g. Neon, Render, or Supabase hosted PostgreSQL)
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable is not set; AI explanations will be unavailable.")
    DB_CONFIG = {"dsn": _DATABASE_URL}
else:
    _REQUIRED_VARS = ("HOST_NAME", "DATABASE_NAME", "USER_NAME", "PASSWORD")
    _missing = [v for v in _REQUIRED_VARS if not os.getenv(v)]
    if _missing:
        logger.warning(
            "Missing database environment variables: %s. Set DATABASE_URL or individual DB parameters.",
            ", ".join(_missing),
        )
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY environment variable is not set; AI explanations will be unavailable.")

    DB_CONFIG = {
        "host": os.getenv("HOST_NAME"),
        "database": os.getenv("DATABASE_NAME"),
        "user": os.getenv("USER_NAME"),
        "password": os.getenv("PASSWORD"),
        "port": int(os.getenv("PORT", "5432")),
    }

TOP_K = int(os.getenv("FLOATCHAT_TOP_K", "3"))
RADIUS_METERS = float(os.getenv("FLOATCHAT_RADIUS_METERS", "50000"))
ENABLE_LIVE_FETCH = os.getenv("ARGO_LIVE_FETCH", "true").lower() not in ("false", "0", "no")
ENABLE_LOCAL_FALLBACK = os.getenv("ARGO_LOCAL_FALLBACK", "true").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# ML models — lazy proxies for fast startup and testing compatibility
# ---------------------------------------------------------------------------
class _LazySentenceTransformer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._real_instance = None

    def _load(self):
        if self._real_instance is None:
            from api.model_manager import get_shared_sentence_transformer
            self._real_instance = get_shared_sentence_transformer(self._model_name)
        return self._real_instance

    def encode(self, *args, **kwargs):
        return self._load().encode(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)


_st_model = _LazySentenceTransformer("all-MiniLM-L6-v2")


class _LazyGeminiModel:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._real_instance = None
        self._configured = False

    def _load(self):
        if self._real_instance is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key and not self._configured:
                genai.configure(api_key=api_key)
                self._configured = True
            logger.info("Using Gemini model: %s", self._model_name)
            self._real_instance = genai.GenerativeModel(self._model_name)
        return self._real_instance

    def generate_content(self, *args, **kwargs):
        return self._load().generate_content(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._load(), name)


_GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
gemini_model = _LazyGeminiModel(_GEMINI_MODEL_NAME)

# Argo provider singleton
_argo_provider = ArgoERDDAPProvider()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Keep these references for test patching compatibility
import psycopg2 as _psycopg2  # noqa: F401 (used in tests via api_module.psycopg2)


# ---------------------------------------------------------------------------
# Legacy helpers (kept for backward-compat and semantic search)
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list:
    res = _st_model.encode(text)
    return res.tolist() if hasattr(res, "tolist") else list(res)


def cosine_similarity(a, b) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def geocode_place(place_name: str):
    """Return (lat, lon) for a place name, or (None, None) on failure."""
    try:
        geolocator = Nominatim(user_agent="floatchat")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
    except Exception as exc:
        logger.warning("Geocoding failed for %r: %s", place_name, exc)
    return None, None


def extract_lat_lon(user_query: str):
    """Parse explicit lat=/long= coordinates from the query string."""
    lat_match = re.search(r"lat\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    lon_match = re.search(r"long\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    if lat_match and lon_match:
        return float(lat_match.group(1)), float(lon_match.group(1))
    return None, None


# ---------------------------------------------------------------------------
# LLM explanation
# ---------------------------------------------------------------------------

def llm_explain(user_query: str, profiles: list[dict]) -> str:
    """
    Generate a conversational, data-grounded explanation via Gemini.
    Never fabricates data — only interprets what is in `profiles`.
    Falls back gracefully on API failure.
    """
    if not profiles:
        return "No matching ocean profiles were found for your query."

    summary_lines = []
    for p in profiles[:3]:  # limit to 3 profiles in prompt
        wmo = p.get("wmo_id", "unknown")
        loc = f"lat={p.get('lat', '?'):.3f}, lon={p.get('lon', '?'):.3f}" if isinstance(p.get("lat"), (int, float)) else "location unknown"
        t = p.get("time", "unknown")
        source = p.get("source", "unknown")
        cache_status = p.get("cache_status", "")
        lvls = p.get("depth_levels", [])[:10]

        header = f"Float WMO {wmo} at {loc} on {t} (source: {source}, {cache_status}):"
        levels_str = "\n".join(
            f"  pres={l.get('pres')} dbar, temp={l.get('temp')}°C, sal={l.get('salinity') or l.get('psal')} PSU"
            for l in lvls
            if l.get("pres") is not None or l.get("temp") is not None
        )
        summary_lines.append(f"{header}\n{levels_str}" if levels_str else header)

    summary = "\n\n".join(summary_lines)
    source_desc = profiles[0].get("source", "database")
    cache_status = profiles[0].get("cache_status", "")

    prompt = (
        f'User asked: "{user_query}"\n\n'
        f"Here is the real ocean data retrieved from {source_desc} ({cache_status}):\n\n"
        f"{summary}\n\n"
        "Based ONLY on the data above (do not invent values), give a clear, concise, "
        "conversational explanation. Include specific numbers from the data. "
        "Mention the float WMO ID, location, and date. Keep it friendly and accurate. "
        "If data is limited, say so honestly. You can use emojis."
    )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        p = profiles[0]
        return (
            f"Found {len(profiles)} profile(s). First: WMO {p.get('wmo_id', '?')} "
            f"at ({p.get('lat', '?'):.3f}, {p.get('lon', '?'):.3f}) "  # type: ignore[arg-type]
            f"on {p.get('time', '?')}. "
            "(AI explanation unavailable — check GOOGLE_API_KEY.)"
        )


# ---------------------------------------------------------------------------

def _answer_educational_query(user_query: str) -> str:
    """
    Provide a concise, scientifically accurate explanation of an oceanographic
    concept without requiring search constraints or fabricating real float data.
    """
    q_lower = user_query.lower()

    curated = [
        (r"\btemp(?:erature)?\b", (
            "Ocean temperature measures the heat content and thermal kinetic energy of seawater, "
            "typically recorded in degrees Celsius (°C). In the ocean, solar radiation warms the surface "
            "layer (0–100 m), below which lies the thermocline where temperatures decline sharply, "
            "eventually reaching 1–4°C in the deep ocean."
        )),
        (r"\bsalinity\b", (
            "Salinity describes the concentration of dissolved mineral salts (primarily sodium and chloride) "
            "in seawater, measured on the Practical Salinity Scale in Practical Salinity Units (PSU). "
            "Global open ocean salinity typically ranges between 33 and 37 PSU, influenced by surface evaporation, "
            "precipitation, ice formation, and river inflows."
        )),
        (r"\bpres(?:sure)?\b", (
            "In oceanography, pressure refers to hydrostatic pressure exerted by the weight of water above a sensor, "
            "conventionally expressed in decibars (dbar). Because seawater density is approximately 1025 kg/m³, "
            "1 dbar of pressure closely corresponds to 1 meter of depth (e.g. 1000 dbar is roughly 1000 meters)."
        )),
        (r"\bargo\s+float\b", (
            "An Argo float is an autonomous, robotic ocean probe that drifts at depths of 1,000 to 2,000 meters. "
            "Every 10 days, it adjusts its buoyancy using an internal hydraulic bladder to rise to the surface, "
            "recording continuous profiles of temperature, salinity, and pressure. Upon surfacing, it transmits "
            "the data via satellite to global data centres before descending to repeat the mission."
        )),
        (r"\bpsu\b", (
            "PSU stands for Practical Salinity Unit, the dimensionless unit of the Practical Salinity Scale (PSS-78). "
            "It is derived from the electrical conductivity ratio of seawater relative to a standard KCl solution at 15°C "
            "and 1 atmosphere. Typical open ocean seawater has a salinity near 35 PSU."
        )),
        (r"\bthermocline\b", (
            "The thermocline is the vertical transition zone in the water column where water temperature "
            "drops significantly more rapidly with increasing depth than in the layers above or below, "
            "acting as a barrier between warm surface waters and cold abyssal depths."
        )),
    ]

    fallback_answer: str | None = None
    for pattern, ans_text in curated:
        if re.search(pattern, q_lower):
            fallback_answer = ans_text
            break

    prompt = (
        f'The user is asking an educational oceanography question: "{user_query}"\n\n'
        "Provide a concise, scientifically accurate, and friendly educational explanation (2-3 paragraphs). "
        "Explain the scientific principles and oceanographic significance (e.g. how it relates to seawater or Argo floats). "
        "Do NOT invent or fabricate specific float numbers, measurements, or coordinates. "
        "Keep the tone informative and accessible, and you may use emojis."
    )
    try:
        response = gemini_model.generate_content(prompt)
        if response and response.text and response.text.strip():
            return response.text.strip()
    except Exception as exc:
        logger.warning("Gemini educational answer failed: %s", exc)

    return fallback_answer or (
        "In oceanography, this concept relates to fundamental physical and chemical properties of seawater "
        "measured continuously by global observation platforms such as the international Argo float array."
    )

# ---------------------------------------------------------------------------
# Numerical calculations
# ---------------------------------------------------------------------------

def compute_stats(profiles: list[dict], variable: str, target_pres: float | None = None) -> dict:
    """
    Compute deterministic statistics from retrieved profile data.
    Used for numerical-answer intent.
    """
    values = []
    for p in profiles:
        for lvl in p.get("depth_levels", []):
            val = lvl.get(variable) or lvl.get({"psal": "salinity"}.get(variable, variable))
            pres = lvl.get("pres")
            if val is None:
                continue
            if target_pres is not None and pres is not None:
                if abs(pres - target_pres) > 50:  # within 50 dbar of target
                    continue
            try:
                v = float(val)
                if math.isfinite(v):
                    values.append(v)
            except (TypeError, ValueError):
                continue

    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(float(np.mean(values)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "std": round(float(np.std(values)), 4),
    }


# ---------------------------------------------------------------------------
# Main query orchestration
# ---------------------------------------------------------------------------

def _get_db_connection(timeout: int | None = None):
    if "dsn" in DB_CONFIG:
        if timeout is not None:
            return psycopg2.connect(DB_CONFIG["dsn"], connect_timeout=timeout)
        return psycopg2.connect(DB_CONFIG["dsn"])
    cfg = dict(DB_CONFIG)
    if timeout is not None:
        cfg["connect_timeout"] = timeout
    return psycopg2.connect(**cfg)


def query_profiles(user_query: str) -> dict:
    """
    Orchestrate the full retrieval pipeline for *user_query*.

    Returns a dict with:
      profiles: list[dict]   — normalised profile records
      source_type: str       — "live" | "cached" | "local" | "empty"
      filters_used: dict     — what filters were parsed
      stats: dict            — numerical stats if applicable
      error: str | None      — retrieval error message if partial failure
    """
    user_query = user_query.strip()
    if not user_query:
        raise ValueError("Empty query")

    parsed = parse_query(user_query)
    logger.info(
        "Parsed query: intent=%s, wmo=%s, region=%s, lat=[%s,%s] lon=[%s,%s] dates=%s/%s",
        parsed.intent,
        parsed.filters.wmo_ids,
        parsed.filters.region_name,
        parsed.filters.min_lat,
        parsed.filters.max_lat,
        parsed.filters.min_lon,
        parsed.filters.max_lon,
        parsed.filters.date_start,
        parsed.filters.date_end,
    )

    retrieval_error: str | None = None
    profiles: list[dict] = []
    source_type = "empty"

    # Educational intent returns pure explanation without database or network retrieval
    if parsed.intent == "educational":
        educational_ans = _answer_educational_query(user_query)
        return {
            "profiles": [],
            "source_type": "educational",
            "filters_used": {},
            "stats": {},
            "retrieval_error": None,
            "ambiguities": [],
            "educational_answer": educational_ans,
        }

    conn = None
    try:
        conn = _get_db_connection()

        # Ensure schema is migrated (idempotent — fast after first run)
        try:
            run_migrations(conn)
        except Exception as exc:
            logger.warning("Schema migration failed (continuing): %s", exc)

        has_structured_filters = (
            parsed.filters.wmo_ids
            or parsed.filters.has_spatial_filter()
            or parsed.filters.has_temporal_filter()
        )

        # ---------------------------------------------------------------
        # Path A: Structured Argo filters present
        # ---------------------------------------------------------------
        if has_structured_filters and parsed.is_argo_query:

            # A1. Cache lookup
            f = parsed.filters
            cached = lookup_cached_profiles(
                conn,
                wmo_ids=f.wmo_ids or None,
                min_lat=f.min_lat,
                max_lat=f.max_lat,
                min_lon=f.min_lon,
                max_lon=f.max_lon,
                date_start=f.date_start,
                date_end=f.date_end,
                max_results=TOP_K * 5,
            )

            # Check freshness for WMO-specific queries
            cache_is_fresh = False
            if f.wmo_ids and cached:
                cache_is_fresh = is_cache_fresh(conn, f.wmo_ids)
            elif cached:
                cache_is_fresh = True  # bbox/date queries use cache as-is

            if cached and (cache_is_fresh or not ENABLE_LIVE_FETCH):
                profiles = cached[:TOP_K]
                source_type = "cached" if profiles[0].get("source") != "local" else "local"
                logger.info("Cache hit: %d profiles", len(profiles))

            # A2. Live fetch if cache miss or stale
            elif ENABLE_LIVE_FETCH:
                logger.info("Cache miss — fetching from ERDDAP")
                try:
                    if f.wmo_ids and len(f.wmo_ids) > 1:
                        raw_profiles = _argo_provider.fetch_multiple_wmo(f)
                    else:
                        raw_profiles = _argo_provider.fetch_profiles(f)

                    if raw_profiles:
                        try:
                            inserted, updated = upsert_profiles(conn, raw_profiles)
                            logger.info("Cached %d new, %d updated profiles", inserted, updated)
                        except Exception as exc:
                            logger.error("Cache write failed: %s", exc)

                        # Re-query from cache to get normalised format
                        profiles = lookup_cached_profiles(
                            conn,
                            wmo_ids=f.wmo_ids or None,
                            min_lat=f.min_lat,
                            max_lat=f.max_lat,
                            min_lon=f.min_lon,
                            max_lon=f.max_lon,
                            date_start=f.date_start,
                            date_end=f.date_end,
                            max_results=TOP_K,
                        )
                        # Tag as live
                        for p in profiles:
                            p["cache_status"] = "live"
                        source_type = "live"
                    else:
                        logger.info("ERDDAP returned 0 profiles")
                        # Fall through to semantic search or return empty

                except ArgoProviderError as exc:
                    retrieval_error = str(exc)
                    logger.warning("ERDDAP retrieval failed: %s", exc)
                    # Fall through to semantic search
                except ValueError as exc:
                    retrieval_error = str(exc)
                    logger.warning("Filter validation error: %s", exc)

        # ---------------------------------------------------------------
        # Path B: Semantic / embedding fallback
        # ---------------------------------------------------------------
        if not profiles and (not has_structured_filters or ENABLE_LOCAL_FALLBACK):
            profiles = _semantic_search(conn, user_query)
            if profiles:
                source_type = "local"
                logger.info("Semantic fallback: %d profiles", len(profiles))

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    # ---------------------------------------------------------------
    # Compute stats for numerical-answer intent
    # ---------------------------------------------------------------
    stats: dict = {}
    if profiles and parsed.intent == "numerical_answer" and parsed.variables:
        target_pres = _extract_target_pressure(user_query)
        for var in parsed.variables[:2]:
            stats[var] = compute_stats(profiles, var, target_pres)

    return {
        "profiles": profiles,
        "source_type": source_type,
        "filters_used": {
            "wmo_ids": parsed.filters.wmo_ids,
            "region": parsed.filters.region_name,
            "min_lat": parsed.filters.min_lat,
            "max_lat": parsed.filters.max_lat,
            "min_lon": parsed.filters.min_lon,
            "max_lon": parsed.filters.max_lon,
            "date_start": parsed.filters.date_start,
            "date_end": parsed.filters.date_end,
        },
        "stats": stats,
        "retrieval_error": retrieval_error,
        "ambiguities": parsed.ambiguities,
    }


def _semantic_search(
    conn: "psycopg2.extensions.connection",
    user_query: str,
) -> list[dict]:
    """
    Embedding-based semantic search over all cached profiles.
    Fallback when no structured filters matched or live fetch failed.
    Checks DB first to avoid loading the embedding model when no embeddings exist.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, latitude, longitude, juld, embedding, source, wmo_id "
            "FROM profiles WHERE embedding IS NOT NULL LIMIT 200"
        )
        rows = cur.fetchall()

    if not rows:
        return []

    # Only load the embedding model and compute query embedding if candidate profiles exist
    query_emb = get_embedding(user_query)

    # Extract point location for proximity scoring
    lat, lon = extract_lat_lon(user_query)
    if lat is None or lon is None:
        lat, lon = geocode_place(user_query)

    sims = []
    for pid, plat, plon, juld, emb, source_val, wmo_id in rows:
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except json.JSONDecodeError:
                continue
        if emb is None:
            continue

        if lat is not None and lon is not None:
            try:
                dist = distance((lat, lon), (plat, plon)).meters
            except Exception:
                dist = 0
            if dist > RADIUS_METERS:
                continue
        else:
            dist = 0

        sim = cosine_similarity(query_emb, emb)
        dist_score = 1.0 / (1.0 + dist)
        score = 0.7 * sim + 0.3 * dist_score
        sims.append((score, pid, plat, plon, juld, source_val, wmo_id))

    if not sims:
        return []

    top = sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_K]

    results = []
    with _get_db_connection() as conn2:
        with conn2.cursor() as cur:
            for _, pid, plat, plon, juld, source_val, wmo_id in top:
                cur.execute(
                    """
                    SELECT pres, temp, psal, pres_qc, temp_qc, psal_qc
                    FROM profile_levels
                    WHERE profile_id = %s
                    ORDER BY n_levels ASC
                    LIMIT 50
                    """,
                    (pid,),
                )
                lvl_rows = cur.fetchall()

                depth_levels = []
                for pres, temp, psal, pres_qc, temp_qc, psal_qc in lvl_rows:
                    if pres is None and temp is None and psal is None:
                        continue
                    try:
                        if any(math.isnan(v) for v in [pres, temp, psal] if v is not None):
                            continue
                    except TypeError:
                        pass
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
                results.append(
                    {
                        "profile_id": pid,
                        "wmo_id": wmo_id,
                        "lat": plat,
                        "lon": plon,
                        "time": juld_str,
                        "source": source_val or "local",
                        "source_url": None,
                        "retrieved_at": None,
                        "data_mode": None,
                        "cache_status": "local",
                        "depth_levels": depth_levels,
                    }
                )

    return results


def _extract_target_pressure(query: str) -> float | None:
    """Extract a target pressure value like '500 dbar' or 'at 200 m'."""
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:dbar|m|db|meter)", query, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _format_response(result: dict) -> list[dict]:
    """
    Convert internal result dict to the frontend-compatible list format.

    Backward-compatible: each item has profile_id, lat, lon, time,
    depth_levels, query_explain. New fields: wmo_id, source, cache_status.
    """
    profiles = result["profiles"]
    if not profiles:
        return []

    user_query = result.get("_user_query", "")
    explain = llm_explain(user_query, profiles) if user_query else ""

    output = []
    for i, p in enumerate(profiles):
        item = {
            # Legacy fields (backward-compatible)
            "profile_id": p.get("profile_id"),
            "lat": p.get("lat"),
            "lon": p.get("lon"),
            "time": p.get("time"),
            "depth_levels": p.get("depth_levels", [])[:15],
            "query_explain": explain if i == 0 else "",
            # New enriched fields
            "wmo_id": p.get("wmo_id"),
            "source": p.get("source", "local"),
            "source_url": p.get("source_url"),
            "cache_status": p.get("cache_status", "local"),
            "data_mode": p.get("data_mode"),
        }
        output.append(item)

    # Attach stats to first item
    if result.get("stats"):
        output[0]["stats"] = result["stats"]

    return output


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "FloatChat API is running 🚀",
            "live_fetch_enabled": ENABLE_LIVE_FETCH,
            "local_fallback_enabled": ENABLE_LOCAL_FALLBACK,
        }
    )


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON with a 'query' field"}), 400

    user_query = data.get("query", "")
    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        return jsonify({"error": "Missing or empty 'query' field"}), 400

    try:
        result = query_profiles(user_query)
        result["_user_query"] = user_query

        if result.get("educational_answer"):
            return jsonify(
                [
                    {
                        "profile_id": None,
                        "lat": None,
                        "lon": None,
                        "time": None,
                        "depth_levels": [],
                        "query_explain": result["educational_answer"],
                        "source": "educational",
                        "cache_status": "none",
                        "wmo_id": None,
                    }
                ]
            )

        if result.get("ambiguities"):
            # Return clarification request
            return jsonify(
                [
                    {
                        "profile_id": None,
                        "lat": None,
                        "lon": None,
                        "time": None,
                        "depth_levels": [],
                        "query_explain": result["ambiguities"][0],
                        "source": "system",
                        "cache_status": "none",
                        "wmo_id": None,
                    }
                ]
            )

        if result.get("retrieval_error") and not result["profiles"]:
            return jsonify(
                [
                    {
                        "profile_id": None,
                        "lat": None,
                        "lon": None,
                        "time": None,
                        "depth_levels": [],
                        "query_explain": (
                            f"⚠️ Remote data retrieval failed: {result['retrieval_error']}. "
                            "Please try again later or refine your query."
                        ),
                        "source": "error",
                        "cache_status": "error",
                        "wmo_id": None,
                    }
                ]
            )

        formatted = _format_response(result)
        return jsonify(formatted)

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except psycopg2.Error as exc:
        logger.error("Database error on /query: %s", exc)
        return jsonify({"error": "Database error — check server logs"}), 500
    except Exception:
        logger.exception("Unexpected error on /query")
        return jsonify({"error": "Internal server error — check server logs"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check with DB connectivity status and fast timeout."""
    db_ok = False
    try:
        conn = _get_db_connection(timeout=3)
        conn.close()
        db_ok = True
    except Exception as exc:
        logger.warning("Health check DB probe failed: %s", exc)
    return jsonify(
        {
            "status": "ok" if db_ok else "degraded",
            "db": "connected" if db_ok else "unreachable",
            "live_fetch": ENABLE_LIVE_FETCH,
        }
    )


@app.route("/refresh", methods=["POST"])
def refresh():
    """
    Force-refresh cached data for specified WMO IDs or filters.
    POST body: {"wmo_ids": ["1234567"], "region": "arabian sea"}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    wmo_ids = data.get("wmo_ids", [])
    region = data.get("region")
    query_str = " ".join(wmo_ids) + (" " + region if region else "")
    if not query_str.strip():
        return jsonify({"error": "Provide wmo_ids or region"}), 400

    parsed = parse_query(query_str)
    if region:
        parsed.filters.region_name = region
        from argo_regions import resolve_region
        bbox = resolve_region(region)
        if bbox:
            parsed.filters.min_lat, parsed.filters.max_lat = bbox[0], bbox[1]
            parsed.filters.min_lon, parsed.filters.max_lon = bbox[2], bbox[3]

    conn = None
    try:
        conn = _get_db_connection()
        raw = _argo_provider.fetch_profiles(parsed.filters)
        if not raw:
            return jsonify({"refreshed": 0, "message": "No profiles found"})
        inserted, updated = upsert_profiles(conn, raw)
        return jsonify({"refreshed": inserted + updated, "inserted": inserted, "updated": updated})
    except ArgoProviderError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception:
        logger.exception("Refresh error")
        return jsonify({"error": "Refresh failed — check server logs"}), 500
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
