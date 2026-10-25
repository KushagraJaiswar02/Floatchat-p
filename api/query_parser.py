"""
api/query_parser.py — Natural language → structured ArgoFilters.

Extracts structured search parameters from user queries without using the LLM
for numerical/filter extraction (deterministic regex + region lookup + validation).

Gemini is consulted only for natural language explanation/interpretation over
retrieved data.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from api.argo_regions import NAMED_REGIONS, resolve_region
from api.argo_provider import ArgoFilters

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger("floatchat.query_parser")

# Default radius in degrees used when user queries "near", "around", or specifies a single point
DEFAULT_SEARCH_RADIUS_DEG = float(os.getenv("FLOATCHAT_SEARCH_RADIUS_DEG", "1.0"))

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Explicit WMO context: "float 6902746", "wmo: 12345", etc.
_WMO_EXPLICIT_RE = re.compile(
    r"\b(?:float|wmo|platform)(?:\s*(?:id|number|#)?)?\s*[:=]?\s*(\d+)\b",
    re.IGNORECASE,
)

# Standalone 7-digit WMO IDs
_WMO_RE = re.compile(r"\b(\d{7})\b")

# Bounding box: bbox(min_lat, max_lat, min_lon, max_lon)
_BBOX_RE = re.compile(
    r"bbox\s*\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,"
    r"\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)",
    re.IGNORECASE,
)

# Range of coordinates: "between latitudes 14 and 16 and longitudes 69 and 71"
_BETWEEN_COORDS_RE = re.compile(
    r"between\s+lat(?:itudes?)?\s+([-+]?\d+\.?\d*)\s*(?:and|to)\s*([-+]?\d+\.?\d*)\s*"
    r"(?:and|,)?\s*long(?:itudes?)?\s+([-+]?\d+\.?\d*)\s*(?:and|to)\s*([-+]?\d+\.?\d*)",
    re.IGNORECASE,
)

# Coordinates with N/S/E/W indicators: e.g. "15°N, 70°E", "15.5 S and 70.2 E"
_NSEW_RE = re.compile(
    r"\b([-+]?\d+\.?\d*)\s*([NSns])\b[^\w\n\r]*[,and\s]*\s*([-+]?\d+\.?\d*)\s*([EWew])\b"
)

# General lat / lon keywords
_LAT_RE = re.compile(
    r"\blat(?:itude)?\s*(?:=|:|\bat\b|\bnear\b|\baround\b|\bof\b|\s+)?\s*([-+]?\d+\.?\d*)\s*([NSns])?",
    re.IGNORECASE,
)
_LON_RE = re.compile(
    r"\blon(?:g(?:itude)?)?\s*(?:=|:|\bat\b|\bnear\b|\baround\b|\bof\b|\s+)?\s*([-+]?\d+\.?\d*)\s*([EWew])?",
    re.IGNORECASE,
)

# Contradictory directional coordinates: e.g. -15°S or -70°W
_CONTRA_DIR_RE = re.compile(r"(-\d+\.?\d*)\s*([NSnsEWew])\b")

# Date patterns
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(202[0-9]|201[0-9])\b")
_BETWEEN_DATE_RE = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:for|in|at|with|$))",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Known oceanographic variables
_VAR_KEYWORDS = {
    "temperature": "temp",
    "temp": "temp",
    "salinity": "psal",
    "psal": "psal",
    "pressure": "pres",
    "pres": "pres",
    "depth": "pres",
    "oxygen": "doxy",
    "doxy": "doxy",
    "o2": "doxy",
}

# Words indicating variable inquiry that might be unsupported
_POTENTIAL_VARS = {
    "ph": "pH is currently not available in this standard Argo dataset.",
    "chlorophyll": "Chlorophyll data is currently not available in this standard Argo dataset.",
    "turbidity": "Turbidity data is currently not available in this standard Argo dataset.",
    "nitrate": "Nitrate data is currently not available in this standard Argo dataset.",
}


# ---------------------------------------------------------------------------
# Public Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """Parsed structured representation of a user's ocean data request."""

    filters: ArgoFilters
    variables: list[str]             # ocean vars of interest
    is_argo_query: bool              # True if query is oceanographic
    intent: str                      # "profile_search" | "numerical_answer" | "comparison"
    raw_query: str
    ambiguities: list[str]           # validation errors or clarification requests


# ---------------------------------------------------------------------------
# Core Parser
# ---------------------------------------------------------------------------

def parse_query(user_query: str) -> ParsedQuery:
    """
    Extract ArgoFilters and intent from a natural-language oceanographic query.

    Handles bounding boxes, coordinate pairs, radius expansions, date ranges,
    named regions, WMO IDs, variable keywords, and explicit validation errors.
    """
    q = user_query.strip()
    q_clean = q.replace("°", " ").replace("\u00b0", " ")
    q_lower = q.lower()

    # --- 1. Check for empty or non-informative input ---
    if not q or not any(c.isalnum() for c in q):
        return ParsedQuery(
            filters=ArgoFilters(),
            variables=[],
            is_argo_query=False,
            intent="profile_search",
            raw_query=q,
            ambiguities=["Please enter an oceanographic query or ask about Argo profiles."],
        )

    # --- 2. Detect oceanographic relevance ---
    # Reject explicit DDL/DML injection keywords immediately
    if re.search(r"\b(drop\s+table|delete\s+from|insert\s+into|update\s+\w+\s+set)\b", q_lower):
        return ParsedQuery(
            filters=ArgoFilters(),
            variables=[],
            is_argo_query=False,
            intent="profile_search",
            raw_query=q,
            ambiguities=["Invalid query: SQL statements are not permitted."],
        )

    ocean_keywords = [
        "ocean", "sea", "argo", "float", "wmo", "profile", "salinity", "temperature",
        "pressure", "psal", "temp", "pres", "depth", "marine", "water", "lat", "lon",
        "latitude", "longitude", "bbox", "current", "thermocline", "pacific", "atlantic",
        "indian", "mediterranean", "bay", "gulf", "basin", "ctd", "psu", "dbar",
    ]
    is_argo = any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in ocean_keywords)

    ambiguities: list[str] = []

    # --- 3. Extract and validate oceanographic variables ---
    variables: list[str] = []
    for kw, var in _VAR_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", q_lower) and var not in variables:
            variables.append(var)

    for unsupp_kw, msg in _POTENTIAL_VARS.items():
        if re.search(r"\b" + re.escape(unsupp_kw) + r"\b", q_lower):
            ambiguities.append(msg)

    # --- 4. Extract and validate WMO IDs ---
    wmo_ids: list[str] = []
    explicit_wmos = _WMO_EXPLICIT_RE.findall(q_clean)
    for w in explicit_wmos:
        if len(w) == 7:
            if w not in wmo_ids:
                wmo_ids.append(w)
        else:
            ambiguities.append(f"Invalid WMO ID '{w}': Argo float WMO IDs must be exactly 7 digits.")

    # Also extract any standalone 7-digit numbers if no explicit WMO was matched
    if not explicit_wmos:
        for w in _WMO_RE.findall(q_clean):
            # Avoid picking up 7-digit numbers that are part of other fields (e.g. coordinates without dot)
            if w not in wmo_ids:
                wmo_ids.append(w)

    # --- 5. Extract and validate geographic coordinates & regions ---
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    region_name: Optional[str] = None

    # Check contradictory direction indicators (e.g. -15°S or -70°W)
    contra_matches = _CONTRA_DIR_RE.findall(q_clean)
    if contra_matches:
        ambiguities.append(
            f"Contradictory coordinate format: negative value with directional indicator '{contra_matches[0][0]} {contra_matches[0][1].upper()}'. Use either negative numbers or N/S/E/W."
        )

    # Check explicit bbox: bbox(min_lat, max_lat, min_lon, max_lon)
    bbox_match = _BBOX_RE.search(q_clean)
    if bbox_match:
        b_min_lat, b_max_lat, b_min_lon, b_max_lon = [float(x) for x in bbox_match.groups()]
        if b_min_lat > b_max_lat:
            ambiguities.append(f"Invalid bounding box: min_lat ({b_min_lat}) is greater than max_lat ({b_max_lat}).")
        elif b_min_lon > b_max_lon:
            ambiguities.append(f"Invalid bounding box: min_lon ({b_min_lon}) is greater than max_lon ({b_max_lon}).")
        elif not (-90 <= b_min_lat <= 90 and -90 <= b_max_lat <= 90):
            ambiguities.append("Latitude coordinates must be between -90 and 90 degrees.")
        elif not (-180 <= b_min_lon <= 180 and -180 <= b_max_lon <= 180):
            ambiguities.append("Longitude coordinates must be between -180 and 180 degrees.")
        else:
            min_lat, max_lat, min_lon, max_lon = b_min_lat, b_max_lat, b_min_lon, b_max_lon

    # Check coordinate range: "between latitudes X and Y and longitudes A and B"
    if min_lat is None and not ambiguities:
        range_match = _BETWEEN_COORDS_RE.search(q_clean)
        if range_match:
            lat1, lat2, lon1, lon2 = [float(x) for x in range_match.groups()]
            r_min_lat, r_max_lat = min(lat1, lat2), max(lat1, lat2)
            r_min_lon, r_max_lon = min(lon1, lon2), max(lon1, lon2)
            if not (-90 <= r_min_lat <= 90 and -90 <= r_max_lat <= 90):
                ambiguities.append("Latitude coordinates must be between -90 and 90 degrees.")
            elif not (-180 <= r_min_lon <= 180 and -180 <= r_max_lon <= 180):
                ambiguities.append("Longitude coordinates must be between -180 and 180 degrees.")
            else:
                min_lat, max_lat, min_lon, max_lon = r_min_lat, r_max_lat, r_min_lon, r_max_lon

    # Check point coordinates with directional indicators or lat/lon keywords
    if min_lat is None and not ambiguities:
        pt_lat: Optional[float] = None
        pt_lon: Optional[float] = None

        # Try N/S/E/W format: "15°N, 70°E" or "15 N and 70 E"
        nsew_match = _NSEW_RE.search(q_clean)
        if nsew_match:
            lat_v, lat_dir, lon_v, lon_dir = nsew_match.groups()
            lat_f = float(lat_v) * (-1 if lat_dir.upper() == "S" else 1)
            lon_f = float(lon_v) * (-1 if lon_dir.upper() == "W" else 1)
            pt_lat, pt_lon = lat_f, lon_f
        else:
            # Try keyword-based lat and lon
            lat_match = _LAT_RE.search(q_clean)
            lon_match = _LON_RE.search(q_clean)

            if lat_match:
                val = float(lat_match.group(1))
                if lat_match.group(2) and lat_match.group(2).upper() == "S":
                    val = -abs(val)
                pt_lat = val

            if lon_match:
                val = float(lon_match.group(1))
                if lon_match.group(2) and lon_match.group(2).upper() == "W":
                    val = -abs(val)
                pt_lon = val

        # Validate point coordinates
        if pt_lat is not None and pt_lon is not None:
            if not (-90 <= pt_lat <= 90):
                ambiguities.append(f"Latitude {pt_lat} is out of range [-90, 90].")
            if not (-180 <= pt_lon <= 180):
                ambiguities.append(f"Longitude {pt_lon} is out of range [-180, 180].")

            if not ambiguities:
                # Apply configurable bounding box radius for point/near queries
                radius = DEFAULT_SEARCH_RADIUS_DEG
                min_lat = max(-90.0, pt_lat - radius)
                max_lat = min(90.0, pt_lat + radius)
                min_lon = max(-180.0, pt_lon - radius)
                max_lon = min(180.0, pt_lon + radius)
                logger.info("Point lat=%s, lon=%s expanded with radius=%.2f deg -> [%s, %s, %s, %s]",
                            pt_lat, pt_lon, radius, min_lat, max_lat, min_lon, max_lon)

        elif pt_lat is not None and pt_lon is None:
            if not (-90 <= pt_lat <= 90):
                ambiguities.append(f"Latitude {pt_lat} is out of range [-90, 90].")
            else:
                ambiguities.append("Latitude was specified without a corresponding longitude. Please provide both coordinates.")

        elif pt_lon is not None and pt_lat is None:
            if not (-180 <= pt_lon <= 180):
                ambiguities.append(f"Longitude {pt_lon} is out of range [-180, 180].")
            else:
                ambiguities.append("Longitude was specified without a corresponding latitude. Please provide both coordinates.")

    # Check named region
    if min_lat is None and not ambiguities:
        # Match against predefined oceanic regions
        for rname in sorted(NAMED_REGIONS.keys(), key=len, reverse=True):
            pattern = r"\b" + re.escape(rname) + r"\b"
            if re.search(pattern, q_lower):
                region_name = rname
                bbox = resolve_region(rname)
                if bbox:
                    min_lat, max_lat, min_lon, max_lon = bbox
                break

    # --- 6. Extract and validate date constraints ---
    date_start: str | None = None
    date_end: str | None = None

    iso_dates = _DATE_ISO_RE.findall(q)
    if iso_dates:
        valid_dates: list[str] = []
        for d in iso_dates:
            try:
                datetime.strptime(d, "%Y-%m-%d")
                valid_dates.append(d)
            except ValueError as exc:
                ambiguities.append(f"Invalid calendar date '{d}': {exc}.")

        if len(valid_dates) >= 2:
            d1 = datetime.strptime(valid_dates[0], "%Y-%m-%d")
            d2 = datetime.strptime(valid_dates[1], "%Y-%m-%d")
            if d1 > d2:
                ambiguities.append(f"Start date ({valid_dates[0]}) cannot be later than end date ({valid_dates[1]}).")
            else:
                date_start, date_end = valid_dates[0], valid_dates[1]
        elif len(valid_dates) == 1:
            date_start = date_end = valid_dates[0]

    if not date_start and not ambiguities:
        # Month-Year extraction
        month_years = _MONTH_YEAR_RE.findall(q)
        if month_years:
            parsed_dates = [_parse_month_year(m, y) for m, y in month_years]
            parsed_dates = [d for d in parsed_dates if d]
            if len(parsed_dates) >= 2:
                d1 = datetime.strptime(parsed_dates[0][0], "%Y-%m-%d")
                d2 = datetime.strptime(parsed_dates[-1][1], "%Y-%m-%d")
                if d1 > d2:
                    ambiguities.append(f"Start date ({parsed_dates[0][0]}) cannot be later than end date ({parsed_dates[-1][1]}).")
                else:
                    date_start, date_end = parsed_dates[0][0], parsed_dates[-1][1]
            elif len(parsed_dates) == 1:
                date_start, date_end = parsed_dates[0][0], parsed_dates[0][1]
        else:
            # Year only
            years = _YEAR_RE.findall(q)
            if years:
                year = min(int(y) for y in years)
                end_year = max(int(y) for y in years)
                date_start = f"{year}-01-01"
                date_end = f"{end_year}-12-31"

    # --- 7. Determine Query Intent ---
    # Distinct categories:
    # 1. "educational": asking for explanations/definitions of ocean concepts
    # 2. "numerical_answer": calculation requests over profiles
    # 3. "comparison": comparing two or more floats/regions
    # 4. "profile_search": general data retrieval request
    retrieval_verbs = ["show", "find", "get", "fetch", "list", "display", "search", "download"]
    has_retrieval_verb = any(re.search(r"\b" + rv + r"\b", q_lower) for rv in retrieval_verbs)

    calc_words = [
        r"\baverage\b", r"\bmean\s+(?:temp|salinity|psal|pressure|pres)\b",
        r"\bmax(?:imum)?\b", r"\bmin(?:imum)?\b", r"\bat\s+depth\b",
        r"\bat\s+pressure\b", r"\bat\s+\d+\s*(?:dbar|m)?\b",
    ]
    has_calc_intent = any(re.search(cw, q_lower) for cw in calc_words)

    edu_pattern = r"^\s*(?:what\s+(?:is|are)(?:\s+(?:a|an|the))?|what\s+does\s+.+?\s+mean|how\s+(?:do|does)\s+.+?\s+work|define|explain|meaning\s+of)\b"
    is_edu_phrase = bool(re.search(edu_pattern, q_lower))

    has_constraints = bool(wmo_ids or min_lat is not None or region_name is not None or date_start)

    if has_calc_intent:
        intent = "numerical_answer"
    elif is_edu_phrase and is_argo and not has_retrieval_verb:
        intent = "educational"
    elif any(w in q_lower for w in ["compare", "comparison", "versus", "vs", "difference"]):
        intent = "comparison"
    else:
        intent = "profile_search"

    # --- 8. Handle missing constraints / clarifications ---
    if is_argo and not ambiguities:
        if intent == "educational":
            # Educational questions do NOT require spatial/temporal/WMO constraints
            pass
        elif not has_constraints:
            if variables:
                var_names = {
                    "temp": "temperature",
                    "psal": "salinity",
                    "pres": "pressure/depth",
                    "doxy": "dissolved oxygen",
                }
                named_vars = [var_names.get(v, v) for v in variables]
                vars_str = " and ".join(named_vars)
                ambiguities.append(
                    f"To retrieve {vars_str} profile data, please specify a location "
                    "(region name, lat/lon, or bounding box), a date range, or a float WMO ID."
                )
            else:
                ambiguities.append(
                    "Please specify a location (region name, lat/lon, or bounding box), "
                    "a date range, or a float WMO ID."
                )

    filters = ArgoFilters(
        wmo_ids=wmo_ids,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        date_start=date_start,
        date_end=date_end,
        region_name=region_name,
        variables=variables,
    )

    return ParsedQuery(
        filters=filters,
        variables=variables,
        is_argo_query=is_argo,
        intent=intent,
        raw_query=q,
        ambiguities=ambiguities,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_month_year(month_str: str, year_str: str) -> tuple[str, str] | None:
    """Return (date_start, date_end) strings for a month+year."""
    month_abbr = month_str[:3].lower()
    month_num = _MONTH_MAP.get(month_abbr)
    if not month_num:
        return None
    year = int(year_str)
    if month_num == 12:
        last_day = 31
    else:
        last_day = (date(year, month_num + 1, 1) - date(year, month_num, 1)).days  # type: ignore[arg-type]
    return (f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day:02d}")
