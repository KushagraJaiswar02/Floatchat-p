"""
api/query_parser.py — Natural language → structured ArgoFilters.

Extracts structured search parameters from user queries without using the LLM
for the numerical/filter extraction (deterministic regex + region lookup).

Gemini is consulted only for intent classification when the query is ambiguous.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from argo_regions import resolve_region
from argo_provider import ArgoFilters

logger = logging.getLogger("floatchat.query_parser")

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# WMO IDs: 7-digit numbers, optionally space/comma separated
_WMO_RE = re.compile(r"\b(\d{7})\b")

# Explicit lat/lon in query
_LAT_RE = re.compile(r"lat(?:itude)?\s*[=:]\s*([-+]?\d+\.?\d*)", re.IGNORECASE)
_LON_RE = re.compile(r"lon(?:g(?:itude)?)?\s*[=:]\s*([-+]?\d+\.?\d*)", re.IGNORECASE)

# Bounding box: bbox(min_lat, max_lat, min_lon, max_lon)
_BBOX_RE = re.compile(
    r"bbox\s*\(\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*,"
    r"\s*([-+]?\d+\.?\d*)\s*,\s*([-+]?\d+\.?\d*)\s*\)",
    re.IGNORECASE,
)

# Date patterns: YYYY-MM-DD or "January 2026", "Jan-Mar 2026"
_DATE_ISO_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?"
    r"\s+(\d{4})",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(202[0-9]|201[0-9])\b")

# Between date range: "between Jan 2026 and Mar 2026"
_BETWEEN_RE = re.compile(
    r"between\s+(.+?)\s+and\s+(.+?)(?:\s+(?:for|in|at|with|$))",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Ocean region keywords that map to named regions
_REGION_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\barabian\s+sea\b", re.IGNORECASE), "arabian sea"),
    (re.compile(r"\bbay\s+of\s+bengal\b", re.IGNORECASE), "bay of bengal"),
    (re.compile(r"\bindian\s+ocean\b", re.IGNORECASE), "indian ocean"),
    (re.compile(r"\bpacific\s+ocean\b", re.IGNORECASE), "pacific ocean"),
    (re.compile(r"\bnorth\s+pacific\b", re.IGNORECASE), "north pacific"),
    (re.compile(r"\bsouth\s+pacific\b", re.IGNORECASE), "south pacific"),
    (re.compile(r"\battlantic\s+ocean\b", re.IGNORECASE), "atlantic ocean"),
    (re.compile(r"\bnorth\s+atlantic\b", re.IGNORECASE), "north atlantic"),
    (re.compile(r"\bsouth\s+atlantic\b", re.IGNORECASE), "south atlantic"),
    (re.compile(r"\bmediterranean\b", re.IGNORECASE), "mediterranean sea"),
    (re.compile(r"\bsouthern\s+ocean\b", re.IGNORECASE), "southern ocean"),
    (re.compile(r"\barctic\s+ocean\b", re.IGNORECASE), "arctic ocean"),
    (re.compile(r"\bred\s+sea\b", re.IGNORECASE), "red sea"),
    (re.compile(r"\bpersian\s+gulf\b", re.IGNORECASE), "persian gulf"),
    (re.compile(r"\bcaribbean\b", re.IGNORECASE), "caribbean sea"),
    (re.compile(r"\bcoral\s+sea\b", re.IGNORECASE), "coral sea"),
    (re.compile(r"\btasman\s+sea\b", re.IGNORECASE), "tasman sea"),
    (re.compile(r"\bsouth\s+china\s+sea\b", re.IGNORECASE), "south china sea"),
]

# Variables of interest
_VAR_KEYWORDS = {
    "temperature": "temp",
    "temp": "temp",
    "salinity": "psal",
    "psal": "psal",
    "pressure": "pres",
    "pres": "pres",
    "depth": "pres",
    "oxygen": "doxy",
    "o2": "doxy",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    """Parsed structured representation of a user's ocean data request."""

    filters: ArgoFilters
    variables: list[str]             # ocean vars of interest
    is_argo_query: bool              # True if query is oceanographic
    intent: str                      # "profile_search" | "numerical_answer" | "comparison"
    raw_query: str
    ambiguities: list[str]           # things the user needs to clarify


def parse_query(user_query: str) -> ParsedQuery:
    """
    Extract ArgoFilters and intent from a natural-language oceanographic query.

    This is purely deterministic (no LLM call). The LLM is used separately for
    explanation / interpretation once data has been retrieved.
    """
    q = user_query.strip()
    q_lower = q.lower()

    # --- Detect if this is an oceanographic query ---
    ocean_keywords = [
        "ocean", "sea", "argo", "float", "wmo", "profile", "salinity", "temperature",
        "pressure", "psal", "temp", "pres", "depth", "marine", "water", "lat", "lon",
        "latitude", "longitude", "bbox", "current", "thermocline",
    ]
    is_argo = any(kw in q_lower for kw in ocean_keywords)

    variables: list[str] = []
    for kw, var in _VAR_KEYWORDS.items():
        if kw in q_lower and var not in variables:
            variables.append(var)

    # --- WMO IDs ---
    wmo_ids = _WMO_RE.findall(q)

    # --- Bounding box ---
    min_lat = max_lat = min_lon = max_lon = None
    bbox_match = _BBOX_RE.search(q)
    if bbox_match:
        min_lat, max_lat, min_lon, max_lon = [float(x) for x in bbox_match.groups()]
    else:
        lat_match = _LAT_RE.search(q)
        lon_match = _LON_RE.search(q)
        if lat_match:
            min_lat = max_lat = float(lat_match.group(1))
        if lon_match:
            min_lon = max_lon = float(lon_match.group(1))

    # --- Named region ---
    region_name: str | None = None
    for pattern, rname in _REGION_KEYWORDS:
        if pattern.search(q):
            region_name = rname
            break

    # --- Dates ---
    date_start: str | None = None
    date_end: str | None = None

    iso_dates = _DATE_ISO_RE.findall(q)
    if len(iso_dates) >= 2:
        dates = sorted(iso_dates)
        date_start, date_end = dates[0], dates[1]
    elif len(iso_dates) == 1:
        date_start = date_end = iso_dates[0]
    else:
        # Try month-year extraction
        month_years = _MONTH_YEAR_RE.findall(q)
        if month_years:
            parsed_dates = [_parse_month_year(m, y) for m, y in month_years]
            parsed_dates = [d for d in parsed_dates if d]
            if len(parsed_dates) >= 2:
                date_start = parsed_dates[0][0]
                date_end = parsed_dates[-1][1]
            elif len(parsed_dates) == 1:
                date_start = parsed_dates[0][0]
                date_end = parsed_dates[0][1]
        else:
            # Year only
            years = _YEAR_RE.findall(q)
            if years:
                year = min(int(y) for y in years)
                end_year = max(int(y) for y in years)
                date_start = f"{year}-01-01"
                date_end = f"{end_year}-12-31"

    # --- Determine intent ---
    intent = "profile_search"
    if any(w in q_lower for w in ["compare", "comparison", "versus", "vs", "difference"]):
        intent = "comparison"
    elif any(w in q_lower for w in ["what is", "calculate", "compute", "average", "mean", "at depth", "at pressure", "at 500"]):
        intent = "numerical_answer"

    # --- Ambiguities ---
    ambiguities: list[str] = []
    if is_argo and not wmo_ids and min_lat is None and region_name is None and not date_start:
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
    # Last day of month (approximate)
    if month_num == 12:
        last_day = 31
    else:
        last_day = (date(year, month_num + 1, 1) - date(year, month_num, 1)).days  # type: ignore[arg-type]
    return (f"{year}-{month_num:02d}-01", f"{year}-{month_num:02d}-{last_day:02d}")
