"""
api/argo_provider.py — Argo ERDDAP data provider (Ifremer GDAC).

Source: https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats
- No API key or authentication required
- Uses CSV format (more reliable than NetCDF for tabledap)
- Bounded by max_profiles; requires at least one filter constraint
- Supports: WMO ID, bounding box, date range, named regions

Column names (all lowercase in ERDDAP ArgoFloats):
  platform_number, time, latitude, longitude, pres, temp, psal,
  pres_qc, temp_qc, psal_qc, data_mode, profile_temp_qc, profile_psal_qc
"""

from __future__ import annotations

from api.telemetry import timed

import logging
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from time import monotonic
from typing import Optional

import pandas as pd
import requests
from urllib3.exceptions import HTTPError as UpstreamHTTPError

from api.argo_regions import resolve_region
from api.query_contract import depth_from_pressure, filter_depth_levels, pressure_at_depth

logger = logging.getLogger("floatchat.argo_provider")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ERDDAP_BASE = os.getenv(
    "ARGO_ERDDAP_BASE",
    "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats",
)
ERDDAP_TIMEOUT = min(float(os.getenv("ARGO_ERDDAP_TIMEOUT", "10")), 12.0)
ERDDAP_CONNECT_TIMEOUT = 3
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_PROFILES = int(os.getenv("ARGO_MAX_PROFILES", "20"))

# Request only required columns to minimise payload
_ERDDAP_VARS = (
    "platform_number",
    "cycle_number",
    "time",
    "latitude",
    "longitude",
    "pres",
    "temp",
    "psal",
    "pres_qc",
    "temp_qc",
    "psal_qc",
    "data_mode",
)

# QC flags considered usable (Argo reference: 1=good, 2=prob good, 8=interpolated)
_GOOD_QC = {"0", "1", "2", "8", " ", ""}


# ---------------------------------------------------------------------------
# Filters dataclass
# ---------------------------------------------------------------------------

@dataclass
class ArgoFilters:
    """Validated search filters for Argo profile retrieval."""

    wmo_ids: list[str] = field(default_factory=list)
    min_lat: Optional[float] = None
    max_lat: Optional[float] = None
    min_lon: Optional[float] = None
    max_lon: Optional[float] = None
    date_start: Optional[str] = None   # "YYYY-MM-DD"
    date_end: Optional[str] = None
    region_name: Optional[str] = None
    variables: list[str] = field(default_factory=list)
    max_profiles: int = MAX_PROFILES
    min_depth_m: Optional[float] = None
    max_depth_m: Optional[float] = None

    def __post_init__(self) -> None:
        if self.region_name:
            bbox = resolve_region(self.region_name)
            if bbox:
                self.min_lat, self.max_lat = bbox[0], bbox[1]
                self.min_lon, self.max_lon = bbox[2], bbox[3]
                logger.debug("Named region resolved")
            else:
                logger.warning("Unknown named region")

    def validate(self) -> list[str]:
        errors: list[str] = []
        for name, val, lo, hi in [
            ("min_lat", self.min_lat, -90, 90),
            ("max_lat", self.max_lat, -90, 90),
            ("min_lon", self.min_lon, -180, 180),
            ("max_lon", self.max_lon, -180, 180),
        ]:
            if val is not None and not (lo <= val <= hi):
                errors.append(f"{name}={val} out of range [{lo}, {hi}]")
        if (
            self.min_lat is not None
            and self.max_lat is not None
            and self.min_lat > self.max_lat
        ):
            errors.append("min_lat > max_lat")
        if not (1 <= self.max_profiles <= 500):
            errors.append("max_profiles must be 1–500")
        for value in (self.min_depth_m, self.max_depth_m):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 12000):
                errors.append("Depth bounds must be finite and within [0, 12000] metres.")
        if self.has_depth_filter() and self.min_depth_m is not None and self.max_depth_m is not None and self.min_depth_m > self.max_depth_m:
            errors.append("Minimum depth exceeds maximum depth.")
        for wmo in self.wmo_ids:
            if not re.fullmatch(r"\d+", wmo):
                errors.append(f"Invalid WMO ID {wmo!r} (must be numeric string)")
        try:
            start = datetime.strptime(self.date_start, "%Y-%m-%d") if self.date_start else None
            end = datetime.strptime(self.date_end, "%Y-%m-%d") if self.date_end else None
            if start and end and start > end:
                errors.append("Start date is after end date.")
        except ValueError:
            errors.append("Invalid calendar date; use YYYY-MM-DD.")
        if self.min_lon is not None and self.max_lon is not None and self.min_lon > self.max_lon:
            errors.append("min_lon > max_lon")
        return errors

    def has_spatial_filter(self) -> bool:
        return any(
            v is not None for v in [self.min_lat, self.max_lat, self.min_lon, self.max_lon]
        )

    def has_temporal_filter(self) -> bool:
        return self.date_start is not None or self.date_end is not None

    def has_depth_filter(self) -> bool:
        return self.min_depth_m is not None or self.max_depth_m is not None


class ProfileBatch(list):
    """A bounded real-data sample; completeness is explicit even for empty lists."""
    def __init__(self, profiles=(), complete=True):
        super().__init__(profiles)
        self.complete = complete


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ArgoERDDAPProvider:
    """
    Retrieves Argo profiles from Ifremer GDAC ERDDAP (public, no auth).

    Protocol:
      1. Build a tabledap CSV URL from filters
      2. GET the CSV (headers = var names, row 2 = units, row 3+ = data)
      3. Group rows by (platform_number, time) to reconstruct profiles
      4. Return normalised list of profile dicts
    """

    @timed("erddap_request")
    def fetch_profiles(self, filters: ArgoFilters) -> list[dict]:
        """
        Fetch profiles matching *filters* from ERDDAP.

        Raises:
            ValueError: invalid filters
            ArgoProviderError: network / server / parsing failure
        """
        errors = filters.validate()
        if errors:
            raise ValueError("Invalid filters: " + "; ".join(errors))

        if (
            not filters.wmo_ids
            and not filters.has_spatial_filter()
            and not filters.has_temporal_filter()
        ):
            raise ValueError(
                "At least one filter required: WMO ID, bounding box, or date range."
            )

        url = self._build_url(filters)
        text = self._request_csv(url)
        return self._filtered_batch(text, url, filters)

    def _filtered_batch(self, text, url, filters):
        parsed = self._parse_csv(text, url)
        # A cap or discarded malformed records prevents an exhaustive-zero claim.
        complete = (getattr(parsed, "complete", True)
                    and max(0, len(text.splitlines()) - 1) < filters.max_profiles * 300
                    and len(parsed) <= filters.max_profiles)
        profiles = []
        for profile in parsed:
            if filters.has_depth_filter() and any(
                depth_from_pressure(level.get("pres"), profile["latitude"]) is None
                for level in profile["depth_levels"]
            ):
                complete = False
            levels = filter_depth_levels(profile["depth_levels"], profile["latitude"],
                                         filters.min_depth_m, filters.max_depth_m)
            if levels:
                profiles.append({**profile, "depth_levels": levels})
        return ProfileBatch(profiles[:filters.max_profiles], complete=complete)

    @timed("erddap_http")
    def _request_csv(self, url: str) -> str:
        # No retries, redirects, pagination, or sequential retry fan-out in /query.
        started = monotonic()
        resp = None
        try:
            resp = requests.get(url, timeout=(ERDDAP_CONNECT_TIMEOUT, ERDDAP_TIMEOUT),
                                stream=True, allow_redirects=False)
            if resp.status_code not in (200, 404):
                raise ArgoProviderError(f"ERDDAP HTTP {resp.status_code}. Please try again later.")
            chunks = []
            size = 0
            while True:
                if monotonic() - started > ERDDAP_CONNECT_TIMEOUT + ERDDAP_TIMEOUT:
                    raise ArgoProviderError("ERDDAP download timed out. Use a shorter date range.")
                # read1 returns available bytes after one socket read, so a slow
                # trickle cannot keep a large iter_content chunk waiting forever.
                chunk = resp.raw.read1(64 * 1024, decode_content=True)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ArgoProviderError("ERDDAP response exceeded the safe size limit. Use a smaller area or date range.")
                chunks.append(chunk)
            text = b"".join(chunks).decode("utf-8")
            if resp.status_code == 404:
                if "no matching results" in text.lower() or "nrows = 0" in text.lower() or "nrows=0" in text.lower():
                    return ""
                raise ArgoProviderError("ERDDAP HTTP 404: dataset or endpoint unavailable; no-match status is unconfirmed.")
            if not text.strip():
                raise ArgoProviderError("ERDDAP returned an empty or incomplete response; no-match status is unconfirmed.")
            return text
        except requests.exceptions.Timeout:
            raise ArgoProviderError(f"ERDDAP timed out (connect limit {ERDDAP_CONNECT_TIMEOUT}s, read limit {ERDDAP_TIMEOUT:g}s). Use a shorter date range.") from None
        except (requests.exceptions.RequestException, UpstreamHTTPError):
            raise ArgoProviderError("Cannot reach ERDDAP or its response was interrupted. Please try again later.") from None
        finally:
            if resp is not None:
                resp.close()

    @timed("erddap_multi_request")
    def fetch_multiple_wmo(self, filters: ArgoFilters) -> list[dict]:
        """
        Fetch multiple WMO IDs using ERDDAP regex constraint.
        Fails promptly if the upstream regex request fails.
        """
        if not filters.wmo_ids:
            return self.fetch_profiles(filters)

        if len(filters.wmo_ids) == 1:
            return self.fetch_profiles(filters)
        errors = filters.validate()
        if errors:
            raise ValueError("Invalid filters: " + "; ".join(errors))

        # Use ERDDAP regex: platform_number=~"^(id1|id2|id3)$"
        regex_filter = ArgoFilters(
            wmo_ids=[],   # handled via regex in URL
            min_lat=filters.min_lat,
            max_lat=filters.max_lat,
            min_lon=filters.min_lon,
            max_lon=filters.max_lon,
            date_start=filters.date_start,
            date_end=filters.date_end,
            max_profiles=filters.max_profiles,
            min_depth_m=filters.min_depth_m,
            max_depth_m=filters.max_depth_m,
        )
        wmo_pattern = "|".join(re.escape(w) for w in filters.wmo_ids)
        url = self._build_url(regex_filter, extra_constraints=[
            f'platform_number=~"^({wmo_pattern})$"'
        ])
        text = self._request_csv(url)
        return self._filtered_batch(text, url, filters)

    # ------------------------------------------------------------------

    def _build_url(
        self,
        filters: ArgoFilters,
        extra_constraints: list[str] | None = None,
    ) -> str:
        """Build ERDDAP tabledap .csvp URL."""
        vars_part = ",".join(_ERDDAP_VARS)
        constraints: list[str] = []

        if filters.wmo_ids and len(filters.wmo_ids) == 1:
            constraints.append(f'platform_number="{filters.wmo_ids[0]}"')

        if filters.min_lat is not None:
            constraints.append(f"latitude>={filters.min_lat}")
        if filters.max_lat is not None:
            constraints.append(f"latitude<={filters.max_lat}")
        if filters.min_lon is not None:
            constraints.append(f"longitude>={filters.min_lon}")
        if filters.max_lon is not None:
            constraints.append(f"longitude<={filters.max_lon}")

        if filters.date_start:
            constraints.append(f'time>="{filters.date_start}T00:00:00Z"')
        if filters.date_end:
            next_day = (datetime.strptime(filters.date_end, "%Y-%m-%d") + timedelta(days=1)).date()
            constraints.append(f'time<"{next_day}T00:00:00Z"')

        # Conservative pressure prefilter over all latitudes, followed by exact
        # latitude-dependent depth filtering on returned measurements.
        if filters.min_depth_m is not None:
            constraints.append(f"pres>={max(0, pressure_at_depth(filters.min_depth_m, 0) - 1e-6)}")
        if filters.max_depth_m is not None:
            constraints.append(f"pres<={pressure_at_depth(filters.max_depth_m, 90) + 1e-6}")

        if extra_constraints:
            constraints.extend(extra_constraints)

        # Bound rows returned per unique (platform, cycle) group
        row_limit = filters.max_profiles * 300   # ~300 levels/profile max
        constraints.append(f'orderByLimit("{row_limit}")')

        constraint_str = "&".join(constraints)
        # Use .csvp: first row is "varname (units)" — simpler to parse
        return f"{ERDDAP_BASE}.csvp?{vars_part}&{constraint_str}"

    @timed("erddap_parse")
    def _parse_csv(self, text: str, source_url: str) -> list[dict]:
        """
        Parse ERDDAP .csvp response.

        .csvp format:
          Row 1: column headers like "platform_number (unitless),time (UTC),..."
          Row 2+: data rows
        """
        if not text.strip():
            return []

        try:
            df = pd.read_csv(StringIO(text), dtype=str)
        except Exception as exc:
            raise ArgoProviderError(f"CSV parse failed: {exc}")

        # Strip units from column names: "temp (degree_C)" → "temp"
        df.columns = [re.sub(r"\s*\(.*?\)", "", c).strip().lower() for c in df.columns]

        required = {"platform_number", "time", "latitude", "longitude"}
        missing = required - set(df.columns)
        if missing:
            raise ArgoProviderError(f"ERDDAP response missing columns: {missing}")
        if df.empty:
            return ProfileBatch()

        retrieved_at = datetime.now(tz=timezone.utc).isoformat()
        profiles: list[dict] = []

        # Group by (platform_number, time) to reconstruct profiles
        group_cols = ["platform_number", "time"]
        for (wmo, t_str), group in df.groupby(group_cols):
            wmo_str = str(wmo).strip()
            if not wmo_str or wmo_str in ("nan", "NaN", ""):
                continue

            row0 = group.iloc[0]
            lat = _safe_float(row0.get("latitude"))
            lon = _safe_float(row0.get("longitude"))
            if lat is None or lon is None:
                continue

            data_mode = str(row0.get("data_mode", "R") or "R").strip()

            # Parse timestamp
            try:
                juld = datetime.fromisoformat(str(t_str).replace("Z", "+00:00"))
            except Exception:
                try:
                    juld = pd.to_datetime(t_str, utc=True).to_pydatetime()
                except Exception:
                    logger.debug("Skipping unparseable time: %s", t_str)
                    continue

            # Build depth levels
            depth_levels: list[dict] = []
            for _, row in group.iterrows():
                pres = _safe_float(row.get("pres"))
                temp = _safe_float(row.get("temp"))
                psal = _safe_float(row.get("psal"))

                if pres is None and temp is None and psal is None:
                    continue

                pres_qc = str(row.get("pres_qc", "0") or "0").strip()
                temp_qc = str(row.get("temp_qc", "0") or "0").strip()
                psal_qc = str(row.get("psal_qc", "0") or "0").strip()

                depth_levels.append(
                    {
                        "pres": pres,
                        "temp": temp,
                        "psal": psal,
                        "salinity": psal,   # alias for frontend compat
                        "pres_qc": pres_qc,
                        "temp_qc": temp_qc,
                        "psal_qc": psal_qc,
                    }
                )

            if not depth_levels:
                continue

            profiles.append(
                {
                    "wmo_id": wmo_str,
                    "juld": juld,
                    "latitude": lat,
                    "longitude": lon,
                    "data_mode": data_mode,
                    "source": "erddap_ifremer",
                    "source_url": source_url,
                    "retrieved_at": retrieved_at,
                    "depth_levels": depth_levels,
                }
            )

        logger.info(
            "Parsed %d profiles from ERDDAP (%d data rows)", len(profiles), len(df)
        )
        return ProfileBatch(profiles, complete=sum(len(p["depth_levels"]) for p in profiles) == len(df))


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class ArgoProviderError(Exception):
    """Network, server, or parsing error in the Argo provider."""


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
