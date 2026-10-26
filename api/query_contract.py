"""Lightweight coordinate, UTC calendar, and depth conventions shared by UI/API."""
import calendar
from datetime import date, datetime, timedelta, timezone
import math
import re

LAT_RE = re.compile(r"\blat(?:itude)?\s*(?:=|:|at\b|near\b|around\b|of\b)?\s*([-+]?\d+(?:\.\d*)?)\s*(?:([NS])\b)?", re.I)
LON_RE = re.compile(r"\blon(?:g(?:itude)?)?\s*(?:=|:|at\b|near\b|around\b|of\b)?\s*([-+]?\d+(?:\.\d*)?)\s*(?:([EW])\b)?", re.I)
PAIR_RE = re.compile(r"(?<!\w)([-+]?\d+(?:\.\d*)?)\s*([NS])\b\s*(?:,|and)?\s*([-+]?\d+(?:\.\d*)?)\s*([EW])\b", re.I)


def has_coordinate_input(text):
    # Route invalid coordinates to the backend too, so validation is visible.
    text = text.replace("°", " ")
    return bool(re.search(r"\b(?:lat|latitude|lon|longitude|bbox)\b", text, re.I)
                or LON_RE.search(text) or PAIR_RE.search(text))


def parse_point_coordinates(text):
    text = text.replace("°", " ")
    pair = PAIR_RE.search(text)
    lat_match, lon_match = LAT_RE.search(text), LON_RE.search(text)
    values = pair.groups() if pair else (
        *(lat_match.groups() if lat_match else (None, None)),
        *(lon_match.groups() if lon_match else (None, None)),
    )
    errors = []
    result = []
    for raw, hemisphere, name, limit in [(*values[:2], "Latitude", 90), (*values[2:], "Longitude", 180)]:
        value = float(raw) if raw is not None else None
        if value is not None and hemisphere:
            if raw.startswith("-"):
                errors.append("Contradictory coordinate format: use signed numbers or hemisphere indicators, not both.")
            value = abs(value) * (-1 if hemisphere.upper() in ("S", "W") else 1)
        if value is not None and not -limit <= value <= limit:
            errors.append(f"{name} {value} is out of range [-{limit}, {limit}].")
        result.append(value)
    lat, lon = result
    if lat is not None and lon is None:
        errors.append("Latitude was specified without a corresponding longitude. Please provide both coordinates.")
    elif lon is not None and lat is None:
        errors.append("Longitude was specified without a corresponding latitude. Please provide both coordinates.")
    return lat, lon, errors


def utc_reference_date(reference_date=None):
    if reference_date is None:
        return datetime.now(timezone.utc).date()
    if isinstance(reference_date, datetime):
        return reference_date.replace(tzinfo=reference_date.tzinfo or timezone.utc).astimezone(timezone.utc).date()
    return reference_date


def relative_dates(text, reference_date=None):
    today = utc_reference_date(reference_date)
    if re.search(r"\blast month\b", text, re.I):
        end = today.replace(day=1) - timedelta(days=1)
        return end.replace(day=1).isoformat(), end.isoformat()
    if re.search(r"\bthis month\b", text, re.I):
        return today.replace(day=1).isoformat(), today.replace(day=calendar.monthrange(today.year, today.month)[1]).isoformat()
    if re.search(r"\blast week\b", text, re.I):
        end = today - timedelta(days=today.weekday() + 1)
        return (end - timedelta(days=6)).isoformat(), end.isoformat()
    return None


def depth_from_pressure(pres, latitude):
    """UNESCO 1983 standard-ocean estimate, metres from measured dbar/latitude."""
    try:
        p, lat = float(pres), float(latitude)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p < 0 or not math.isfinite(lat) or not -90 <= lat <= 90:
        return None
    x = math.sin(math.radians(lat)) ** 2
    gravity = 9.780318 * (1 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * p
    return (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p / gravity


def depth_sql(p="l.pres", lat="profiles.latitude"):
    """Same UNESCO estimate for SQL filtering BEFORE profile/level LIMITs.

    Arguments are internal column expressions, never user input.
    """
    x = f"power(sin(radians({lat})), 2)"
    return (f"((((-1.82e-15*{p}+2.279e-10)*{p}-2.2512e-5)*{p}+9.72659)*{p}) / "
            f"(9.780318*(1+(5.2788e-3+2.36e-5*{x})*{x})+1.092e-6*{p})")


def pressure_at_depth(depth, latitude):
    lo, hi = 0.0, 15000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if depth_from_pressure(mid, latitude) < depth:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def filter_depth_levels(levels, latitude, minimum=None, maximum=None):
    output = []
    for level in levels:
        depth = depth_from_pressure(level.get("pres"), latitude)
        if minimum is not None or maximum is not None:
            if depth is None or (minimum is not None and depth < minimum) or (maximum is not None and depth > maximum):
                continue
        output.append({**level, "depth_m": depth})
    return output
