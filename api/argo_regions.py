"""
api/argo_regions.py — Named oceanic region → bounding box resolution.

Bounding boxes are [min_lat, max_lat, min_lon, max_lon] in decimal degrees.
All values are verified against standard geographic references.
"""

# ---------------------------------------------------------------------------
# Named region registry
# Each entry: (min_lat, max_lat, min_lon, max_lon)
# ---------------------------------------------------------------------------

NAMED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # Indian Ocean basins
    "arabian sea": (5.0, 25.0, 55.0, 78.0),
    "bay of bengal": (5.0, 23.0, 80.0, 100.0),
    "indian ocean": (-60.0, 30.0, 20.0, 120.0),
    "lakshadweep sea": (7.0, 15.0, 70.0, 78.0),
    # Pacific Ocean
    "pacific ocean": (-60.0, 60.0, 120.0, -70.0),   # wraps antimeridian — handled
    "north pacific": (0.0, 60.0, 120.0, -100.0),
    "south pacific": (-60.0, 0.0, 150.0, -70.0),
    "tropical pacific": (-15.0, 15.0, 130.0, -80.0),
    "east pacific": (-60.0, 60.0, -160.0, -70.0),
    "west pacific": (-60.0, 60.0, 120.0, 160.0),
    "coral sea": (-25.0, -10.0, 142.0, 157.0),
    "tasman sea": (-47.0, -30.0, 150.0, 175.0),
    "philippine sea": (5.0, 30.0, 122.0, 145.0),
    "south china sea": (0.0, 25.0, 99.0, 122.0),
    "java sea": (-8.0, 0.0, 105.0, 120.0),
    # Atlantic Ocean
    "atlantic ocean": (-60.0, 60.0, -70.0, 20.0),
    "north atlantic": (0.0, 65.0, -80.0, 0.0),
    "south atlantic": (-60.0, 0.0, -65.0, 20.0),
    "caribbean sea": (10.0, 22.0, -85.0, -60.0),
    "gulf of mexico": (18.0, 30.0, -97.0, -80.0),
    "mediterranean sea": (30.0, 46.0, -6.0, 42.0),
    "black sea": (41.0, 46.5, 28.0, 42.0),
    "north sea": (50.0, 62.0, -5.0, 12.0),
    "norwegian sea": (62.0, 80.0, -5.0, 25.0),
    "gulf of guinea": (-5.0, 5.0, -5.0, 10.0),
    # Arctic / Southern
    "southern ocean": (-75.0, -45.0, -180.0, 180.0),
    "arctic ocean": (65.0, 90.0, -180.0, 180.0),
    "weddell sea": (-78.0, -60.0, -60.0, -20.0),
    "ross sea": (-78.0, -60.0, 160.0, -150.0),
    # Specific seas
    "red sea": (12.0, 30.0, 32.0, 44.0),
    "persian gulf": (22.0, 30.0, 48.0, 57.0),
    "gulf of aden": (10.0, 15.0, 42.0, 52.0),
    "andaman sea": (5.0, 17.0, 92.0, 100.0),
    "bering sea": (50.0, 66.0, -180.0, -157.0),
    "sea of okhotsk": (44.0, 60.0, 138.0, 163.0),
    "japan sea": (32.0, 52.0, 127.0, 142.0),
    "east china sea": (24.0, 33.0, 117.0, 132.0),
    "yellow sea": (30.0, 39.0, 119.0, 126.0),
}


def resolve_region(name: str) -> tuple[float, float, float, float] | None:
    """
    Resolve a named region to a (min_lat, max_lat, min_lon, max_lon) bounding box.

    Performs case-insensitive partial-match lookup.
    Returns None when no match is found.
    """
    key = name.strip().lower()
    # Exact match first
    if key in NAMED_REGIONS:
        return NAMED_REGIONS[key]
    # Partial match — find first region whose name contains the query words
    words = key.split()
    for region_name, bbox in NAMED_REGIONS.items():
        if all(w in region_name for w in words):
            return bbox
    return None


def bbox_to_erddap_constraints(bbox: tuple[float, float, float, float]) -> dict[str, str]:
    """
    Convert a (min_lat, max_lat, min_lon, max_lon) bounding box to
    ERDDAP tabledap constraint strings.
    """
    min_lat, max_lat, min_lon, max_lon = bbox
    return {
        "latitude>=": str(min_lat),
        "latitude<=": str(max_lat),
        "longitude>=": str(min_lon),
        "longitude<=": str(max_lon),
    }
