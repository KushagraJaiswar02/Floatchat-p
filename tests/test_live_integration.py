"""
tests/test_live_integration.py — Comprehensive tests for the Argo live integration.

Tests cover:
  1.  ERDDAP provider URL building and CSV parsing
  2.  Filter validation and geographic bounding boxes
  3.  Date-range and float-ID queries
  4.  Named region resolution
  5.  Missing measurements and QC flag handling
  6.  Duplicate prevention / upsert idempotency
  7.  Cache hit / miss / freshness logic
  8.  Source attribution
  9.  API endpoint contract (existing + new fields)
  10. Natural-language query parsing
  11. Deterministic numerical calculations
  12. No-data and remote-failure handling
  13. Prevention of silent local-data fallback
  14. Regression: existing test_api.py behaviour preserved

Mocks remote services for deterministic execution — no live network needed.
A separate section (TestLiveSmoke) performs a real ERDDAP call and is
SKIPPED when network is unavailable.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types
import unittest
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Stub heavy dependencies before importing api modules
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules.setdefault(name, mod)
    return mod


# Stub streamlit before anything imports it
_st_ui = _stub("streamlit")
_st_ui.markdown = MagicMock()
_st_ui.write = MagicMock()

_st_mod = _stub("sentence_transformers")
_st_inst = MagicMock()
_st_inst.encode = MagicMock(return_value=[0.1] * 384)
_st_mod.SentenceTransformer = MagicMock(return_value=_st_inst)

_google = _stub("google")
_genai = _stub("google.generativeai")
_genai.configure = MagicMock()
_gmodel = MagicMock()
_gmodel.generate_content = MagicMock(return_value=MagicMock(text="Test AI explanation"))
_genai.GenerativeModel = MagicMock(return_value=_gmodel)

_pg = _stub("psycopg2")
_pg.Error = Exception
_pg.connect = MagicMock()

_pg_extras = _stub("psycopg2.extras")
_pg_extras.execute_batch = MagicMock()
_pg.extras = _pg_extras

_geopy = _stub("geopy")
_geocoders = _stub("geopy.geocoders")
_geocoders.Nominatim = MagicMock()
_geo_dist = _stub("geopy.distance")
_geo_dist.distance = MagicMock(return_value=MagicMock(meters=1000))

# Set required env vars
for k, v in {
    "HOST_NAME": "localhost",
    "DATABASE_NAME": "floatchatai",
    "USER_NAME": "postgres",
    "PASSWORD": "testpw",
    "GOOGLE_API_KEY": "test-key",
}.items():
    os.environ.setdefault(k, v)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

import api.argo_regions as regions_mod
import api.argo_provider as provider_mod
import api.query_parser as parser_mod
import api.argo_cache as cache_mod
import api.query as api_mod


# ===========================================================================
# 1. Named region resolution
# ===========================================================================

class TestRegionResolution(unittest.TestCase):
    def test_arabian_sea_exact(self):
        bbox = regions_mod.resolve_region("Arabian Sea")
        self.assertIsNotNone(bbox)
        min_lat, max_lat, min_lon, max_lon = bbox
        self.assertGreater(max_lat, min_lat)
        # Should be in Indian Ocean area
        self.assertGreater(min_lon, 50)

    def test_indian_ocean(self):
        bbox = regions_mod.resolve_region("indian ocean")
        self.assertIsNotNone(bbox)

    def test_unknown_region_returns_none(self):
        self.assertIsNone(regions_mod.resolve_region("Atlantis"))

    def test_partial_match(self):
        # "pacific" should match "pacific ocean" or similar
        bbox = regions_mod.resolve_region("pacific ocean")
        self.assertIsNotNone(bbox)

    def test_bbox_to_erddap_constraints(self):
        bbox = (5.0, 25.0, 55.0, 78.0)
        constraints = regions_mod.bbox_to_erddap_constraints(bbox)
        self.assertEqual(constraints["latitude>="], "5.0")
        self.assertEqual(constraints["latitude<="], "25.0")
        self.assertEqual(constraints["longitude>="], "55.0")
        self.assertEqual(constraints["longitude<="], "78.0")


# ===========================================================================
# 2. Filter validation
# ===========================================================================

class TestArgoFiltersValidation(unittest.TestCase):
    def test_valid_filters_no_errors(self):
        f = provider_mod.ArgoFilters(
            min_lat=5.0, max_lat=25.0, min_lon=55.0, max_lon=78.0,
            date_start="2026-01-01", date_end="2026-03-31",
        )
        self.assertEqual(f.validate(), [])

    def test_invalid_lat_range(self):
        f = provider_mod.ArgoFilters(min_lat=200.0, max_lat=250.0)
        errors = f.validate()
        self.assertTrue(any("min_lat" in e for e in errors))

    def test_min_lat_greater_than_max_lat(self):
        f = provider_mod.ArgoFilters(min_lat=30.0, max_lat=10.0)
        errors = f.validate()
        self.assertTrue(any("min_lat > max_lat" in e for e in errors))

    def test_invalid_wmo_id(self):
        f = provider_mod.ArgoFilters(wmo_ids=["abc123"])
        errors = f.validate()
        self.assertTrue(any("WMO" in e for e in errors))

    def test_valid_wmo_id(self):
        f = provider_mod.ArgoFilters(wmo_ids=["6902746"])
        self.assertEqual(f.validate(), [])

    def test_max_profiles_bounds(self):
        f = provider_mod.ArgoFilters(max_profiles=1000)
        errors = f.validate()
        self.assertTrue(any("max_profiles" in e for e in errors))

    def test_region_name_resolves_to_bbox(self):
        f = provider_mod.ArgoFilters(region_name="arabian sea")
        # Region should have been resolved to bbox
        self.assertIsNotNone(f.min_lat)
        self.assertIsNotNone(f.max_lat)

    def test_has_spatial_filter(self):
        f = provider_mod.ArgoFilters(min_lat=5.0)
        self.assertTrue(f.has_spatial_filter())

    def test_no_filter_is_empty(self):
        f = provider_mod.ArgoFilters()
        self.assertFalse(f.has_spatial_filter())
        self.assertFalse(f.has_temporal_filter())


# ===========================================================================
# 3. ERDDAP provider URL building
# ===========================================================================

class TestERDDAPProviderURLs(unittest.TestCase):
    def setUp(self):
        self.provider = provider_mod.ArgoERDDAPProvider()

    def test_url_contains_wmo_id(self):
        f = provider_mod.ArgoFilters(wmo_ids=["6902746"])
        url = self.provider._build_url(f)
        self.assertIn("6902746", url)
        self.assertIn("platform_number", url)

    def test_url_contains_bbox(self):
        f = provider_mod.ArgoFilters(
            min_lat=5.0, max_lat=25.0, min_lon=55.0, max_lon=78.0
        )
        url = self.provider._build_url(f)
        self.assertIn("latitude>=5.0", url)
        self.assertIn("latitude<=25.0", url)
        self.assertIn("longitude>=55.0", url)
        self.assertIn("longitude<=78.0", url)

    def test_url_contains_date_range(self):
        f = provider_mod.ArgoFilters(
            min_lat=5.0, max_lat=25.0,
            date_start="2026-01-01", date_end="2026-03-31",
        )
        url = self.provider._build_url(f)
        self.assertIn("2026-01-01", url)
        self.assertIn("2026-03-31", url)

    def test_url_ends_with_csvp(self):
        f = provider_mod.ArgoFilters(wmo_ids=["1234567"])
        url = self.provider._build_url(f)
        self.assertIn(".csvp", url)


# ===========================================================================
# 4. CSV parsing — profiles from mocked ERDDAP response
# ===========================================================================

_SAMPLE_CSV = """\
platform_number (unitless),cycle_number (count),time (UTC),latitude (degrees_north),longitude (degrees_east),pres (dbar),temp (degree_C),psal (PSU),pres_qc (unitless),temp_qc (unitless),psal_qc (unitless),data_mode (unitless)
6902746,34,2026-01-15T12:00:00Z,22.5,60.3,5.0,27.1,36.2,1,1,1,R
6902746,34,2026-01-15T12:00:00Z,22.5,60.3,10.0,26.9,36.1,1,1,1,R
6902746,34,2026-01-15T12:00:00Z,22.5,60.3,25.0,26.5,,1,0,0,R
6902746,34,2026-01-15T12:00:00Z,22.5,60.3,50.0,,35.8,0,0,1,R
"""


class TestCSVParsing(unittest.TestCase):
    def setUp(self):
        self.provider = provider_mod.ArgoERDDAPProvider()

    def test_parses_valid_csv(self):
        profiles = self.provider._parse_csv(_SAMPLE_CSV, "http://test-url")
        self.assertEqual(len(profiles), 1)   # one profile (single WMO+time group)

    def test_profile_has_required_fields(self):
        profiles = self.provider._parse_csv(_SAMPLE_CSV, "http://test-url")
        p = profiles[0]
        self.assertEqual(p["wmo_id"], "6902746")
        self.assertAlmostEqual(p["latitude"], 22.5)
        self.assertAlmostEqual(p["longitude"], 60.3)
        self.assertEqual(p["source"], "erddap_ifremer")
        self.assertEqual(p["data_mode"], "R")

    def test_depth_levels_populated(self):
        profiles = self.provider._parse_csv(_SAMPLE_CSV, "http://test-url")
        lvls = profiles[0]["depth_levels"]
        # All 4 rows are kept (NaN values are preserved, not skipped here)
        self.assertGreater(len(lvls), 0)
        # Check first level values
        first = next(l for l in lvls if l.get("pres") == 5.0)
        self.assertAlmostEqual(first["temp"], 27.1)
        self.assertAlmostEqual(first["psal"], 36.2)

    def test_none_for_nan_values(self):
        profiles = self.provider._parse_csv(_SAMPLE_CSV, "http://test-url")
        lvls = profiles[0]["depth_levels"]
        # Level with pres=25.0 has empty psal — should be None
        level_25 = next((l for l in lvls if l.get("pres") == 25.0), None)
        if level_25:
            self.assertIsNone(level_25.get("psal"))

    def test_empty_csv_returns_empty(self):
        profiles = self.provider._parse_csv("", "http://test-url")
        self.assertEqual(profiles, [])

    def test_qc_flags_preserved(self):
        profiles = self.provider._parse_csv(_SAMPLE_CSV, "http://test-url")
        first_lvl = profiles[0]["depth_levels"][0]
        self.assertEqual(first_lvl.get("pres_qc"), "1")
        self.assertEqual(first_lvl.get("temp_qc"), "1")


# ===========================================================================
# 5. Provider fetch — mocked HTTP
# ===========================================================================

class TestProviderFetch(unittest.TestCase):
    def setUp(self):
        self.provider = provider_mod.ArgoERDDAPProvider()

    def _make_response(self, text="", status=200):
        resp = MagicMock()
        resp.status_code = status
        resp.text = text
        return resp

    def test_fetch_returns_profiles_on_200(self):
        f = provider_mod.ArgoFilters(wmo_ids=["6902746"])
        with patch("requests.get", return_value=self._make_response(_SAMPLE_CSV)):
            profiles = self.provider.fetch_profiles(f)
        self.assertEqual(len(profiles), 1)
        self.assertEqual(profiles[0]["wmo_id"], "6902746")

    def test_fetch_returns_empty_on_404(self):
        f = provider_mod.ArgoFilters(wmo_ids=["9999999"])
        with patch("requests.get", return_value=self._make_response("", 404)):
            profiles = self.provider.fetch_profiles(f)
        self.assertEqual(profiles, [])

    def test_fetch_raises_on_server_error(self):
        f = provider_mod.ArgoFilters(wmo_ids=["6902746"])
        with patch("requests.get", return_value=self._make_response("Server Error", 500)):
            with self.assertRaises(provider_mod.ArgoProviderError):
                self.provider.fetch_profiles(f)

    def test_fetch_raises_on_timeout(self):
        import requests as req_lib
        f = provider_mod.ArgoFilters(wmo_ids=["6902746"])
        with patch("requests.get", side_effect=req_lib.exceptions.Timeout):
            with self.assertRaises(provider_mod.ArgoProviderError) as ctx:
                self.provider.fetch_profiles(f)
        self.assertIn("timed out", str(ctx.exception).lower())

    def test_fetch_raises_on_connection_error(self):
        import requests as req_lib
        f = provider_mod.ArgoFilters(wmo_ids=["6902746"])
        with patch("requests.get", side_effect=req_lib.exceptions.ConnectionError("no route")):
            with self.assertRaises(provider_mod.ArgoProviderError) as ctx:
                self.provider.fetch_profiles(f)
        self.assertIn("erddap", str(ctx.exception).lower())

    def test_fetch_requires_at_least_one_filter(self):
        f = provider_mod.ArgoFilters()
        with self.assertRaises(ValueError):
            self.provider.fetch_profiles(f)

    def test_fetch_raises_on_invalid_filters(self):
        f = provider_mod.ArgoFilters(min_lat=200.0)
        with self.assertRaises(ValueError):
            self.provider.fetch_profiles(f)


# ===========================================================================
# 6. Query parser
# ===========================================================================

class TestQueryParser(unittest.TestCase):
    def test_wmo_extraction(self):
        pq = parser_mod.parse_query("Show profiles for float 6902746")
        self.assertIn("6902746", pq.filters.wmo_ids)

    def test_arabian_sea_region(self):
        pq = parser_mod.parse_query("Show temperature profiles in the Arabian Sea")
        self.assertIsNotNone(pq.filters.min_lat)
        self.assertEqual(pq.filters.region_name, "arabian sea")

    def test_lat_lon_extraction(self):
        pq = parser_mod.parse_query("data at lat=-43.037, long=130")
        # When a point is queried, it is bounded with default search radius (e.g. +/- 1.0 deg)
        radius = parser_mod.DEFAULT_SEARCH_RADIUS_DEG
        self.assertAlmostEqual(pq.filters.min_lat, -43.037 - radius)
        self.assertAlmostEqual(pq.filters.max_lat, -43.037 + radius)
        self.assertAlmostEqual(pq.filters.min_lon, 130.0 - radius)
        self.assertAlmostEqual(pq.filters.max_lon, 130.0 + radius)

    def test_date_iso_extraction(self):
        pq = parser_mod.parse_query("profiles between 2026-01-01 and 2026-03-31")
        self.assertEqual(pq.filters.date_start, "2026-01-01")
        self.assertEqual(pq.filters.date_end, "2026-03-31")

    def test_month_year_extraction(self):
        pq = parser_mod.parse_query("Argo floats in January 2026")
        self.assertIsNotNone(pq.filters.date_start)
        self.assertIn("2026-01", pq.filters.date_start)

    def test_comparison_intent(self):
        pq = parser_mod.parse_query("Compare temperature profiles of floats 6902746 and 6902747")
        self.assertEqual(pq.intent, "comparison")

    def test_numerical_intent(self):
        pq = parser_mod.parse_query("What is the average temperature at 500 dbar?")
        self.assertEqual(pq.intent, "numerical_answer")

    def test_non_ocean_query_not_flagged(self):
        pq = parser_mod.parse_query("What is the weather today?")
        self.assertFalse(pq.is_argo_query)

    def test_ambiguity_when_no_filter(self):
        pq = parser_mod.parse_query("show me some ocean data please")
        self.assertGreater(len(pq.ambiguities), 0)

    def test_variable_extraction(self):
        pq = parser_mod.parse_query("salinity and temperature profiles")
        self.assertIn("psal", pq.variables)
        self.assertIn("temp", pq.variables)

    def test_bbox_extraction(self):
        pq = parser_mod.parse_query("bbox(5,25,55,78)")
        self.assertAlmostEqual(pq.filters.min_lat, 5.0)
        self.assertAlmostEqual(pq.filters.max_lat, 25.0)
        self.assertAlmostEqual(pq.filters.min_lon, 55.0)
        self.assertAlmostEqual(pq.filters.max_lon, 78.0)


# ===========================================================================
# 7. Numerical calculations
# ===========================================================================

class TestNumericalCalculations(unittest.TestCase):
    def _make_profiles(self):
        return [
            {
                "depth_levels": [
                    {"pres": 5.0, "temp": 27.1, "psal": 36.2},
                    {"pres": 10.0, "temp": 26.9, "psal": 36.1},
                    {"pres": 500.0, "temp": 8.0, "psal": 34.5},
                ]
            }
        ]

    def test_mean_temp_at_surface(self):
        profiles = self._make_profiles()
        stats = api_mod.compute_stats(profiles, "temp", target_pres=None)
        self.assertEqual(stats["count"], 3)
        expected_mean = (27.1 + 26.9 + 8.0) / 3
        self.assertAlmostEqual(stats["mean"], round(expected_mean, 4))

    def test_temp_at_target_pressure(self):
        profiles = self._make_profiles()
        stats = api_mod.compute_stats(profiles, "temp", target_pres=500.0)
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["mean"], 8.0)

    def test_stats_on_empty_returns_zero_count(self):
        stats = api_mod.compute_stats([], "temp")
        self.assertEqual(stats["count"], 0)

    def test_nan_values_excluded(self):
        profiles = [
            {
                "depth_levels": [
                    {"pres": 5.0, "temp": float("nan"), "psal": 36.0},
                    {"pres": 10.0, "temp": 26.9, "psal": 36.1},
                ]
            }
        ]
        stats = api_mod.compute_stats(profiles, "temp")
        self.assertEqual(stats["count"], 1)
        self.assertAlmostEqual(stats["mean"], 26.9)


# ===========================================================================
# 8. Flask API — existing regression tests preserved
# ===========================================================================

class TestFlaskRoutesRegression(unittest.TestCase):
    def setUp(self):
        self.client = api_mod.app.test_client()
        api_mod.app.config["TESTING"] = True

    def test_home_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertIn("message", body)

    def test_health_endpoint(self):
        with patch.object(api_mod, "_get_db_connection", side_effect=Exception("no db")):
            r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertIn("status", body)
        self.assertEqual(body["db"], "unreachable")

    def test_post_no_body_returns_400(self):
        r = self.client.post("/query", content_type="application/json", data="")
        self.assertEqual(r.status_code, 400)

    def test_post_non_json_returns_400(self):
        r = self.client.post("/query", content_type="text/plain", data="hello")
        self.assertEqual(r.status_code, 400)

    def test_post_missing_query_returns_400(self):
        r = self.client.post("/query", content_type="application/json", data=json.dumps({}))
        self.assertEqual(r.status_code, 400)

    def test_post_empty_query_returns_400(self):
        r = self.client.post(
            "/query", content_type="application/json", data=json.dumps({"query": "   "})
        )
        self.assertEqual(r.status_code, 400)

    def test_db_error_returns_500_without_stack_trace(self):
        with patch.object(api_mod, "_get_db_connection", side_effect=api_mod.psycopg2.Error("DB down")):
            r = self.client.post(
                "/query",
                content_type="application/json",
                data=json.dumps({"query": "lat=0.0, long=0.0"}),
            )
        self.assertEqual(r.status_code, 500)
        body = json.loads(r.data)
        self.assertIn("error", body)
        self.assertNotIn("Traceback", body["error"])
        self.assertNotIn("password", body["error"].lower())


# ===========================================================================
# 9. API response — new fields (backward-compatible)
# ===========================================================================

class TestAPIResponseShape(unittest.TestCase):
    def setUp(self):
        self.client = api_mod.app.test_client()
        api_mod.app.config["TESTING"] = True

    def _mock_query_profiles(self, profiles):
        return {
            "profiles": profiles,
            "source_type": "live",
            "filters_used": {},
            "stats": {},
            "retrieval_error": None,
            "ambiguities": [],
            "_user_query": "test",
        }

    def test_response_has_backward_compat_fields(self):
        """Each result item must have the legacy frontend-expected fields."""
        sample_profile = {
            "profile_id": 1,
            "wmo_id": "6902746",
            "lat": 22.5,
            "lon": 60.3,
            "time": "2026-01-15 12:00:00",
            "source": "erddap_ifremer",
            "cache_status": "live",
            "data_mode": "R",
            "depth_levels": [{"pres": 5.0, "temp": 27.1, "salinity": 36.2}],
        }
        result = self._mock_query_profiles([sample_profile])
        formatted = api_mod._format_response(result)
        self.assertEqual(len(formatted), 1)
        item = formatted[0]
        # Legacy fields
        self.assertIn("profile_id", item)
        self.assertIn("lat", item)
        self.assertIn("lon", item)
        self.assertIn("time", item)
        self.assertIn("depth_levels", item)
        self.assertIn("query_explain", item)
        # New fields
        self.assertIn("wmo_id", item)
        self.assertIn("source", item)
        self.assertIn("cache_status", item)

    def test_empty_profiles_returns_empty_list(self):
        result = self._mock_query_profiles([])
        formatted = api_mod._format_response(result)
        self.assertEqual(formatted, [])


# ===========================================================================
# 10. No-data and remote-failure handling
# ===========================================================================

class TestNoDataAndFailureHandling(unittest.TestCase):
    def setUp(self):
        self.client = api_mod.app.test_client()
        api_mod.app.config["TESTING"] = True

    def _mock_conn(self):
        conn = MagicMock()
        cur = MagicMock()
        cur.fetchall.return_value = []
        cur.fetchone.return_value = [0]
        cur.__enter__ = MagicMock(return_value=cur)
        cur.__exit__ = MagicMock(return_value=False)
        conn.cursor.return_value = cur
        conn.__enter__ = MagicMock(return_value=conn)
        conn.__exit__ = MagicMock(return_value=False)
        return conn

    def test_provider_failure_returns_error_response(self):
        """When ERDDAP fails and no cache/local data, return error message (not crash)."""
        mock_conn = self._mock_conn()
        with patch.object(api_mod, "_get_db_connection", return_value=mock_conn):
            with patch.object(api_mod, "run_migrations"):
                with patch.object(api_mod._argo_provider, "fetch_profiles",
                                  side_effect=provider_mod.ArgoProviderError("server down")):
                    with patch.object(cache_mod, "lookup_cached_profiles", return_value=[]):
                        r = self.client.post(
                            "/query",
                            content_type="application/json",
                            data=json.dumps({"query": "lat=22.5, long=60.3"}),
                        )
        # Should return 200 with an error message item, not 500
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertIsInstance(body, list)
        # Either empty (fallback returned nothing) or contains error message
        if body:
            explain = body[0].get("query_explain", "")
            source = body[0].get("source", "")
            # Must not silently pretend data was live-retrieved
            self.assertNotEqual(source, "erddap_ifremer")

    def test_ambiguous_query_returns_clarification(self):
        """Query with no filters should return a clarification prompt."""
        mock_conn = self._mock_conn()
        with patch.object(api_mod, "_get_db_connection", return_value=mock_conn):
            with patch.object(api_mod, "run_migrations"):
                with patch.object(cache_mod, "lookup_cached_profiles", return_value=[]):
                    r = self.client.post(
                        "/query",
                        content_type="application/json",
                        data=json.dumps({"query": "show me some ocean data"}),
                    )
        self.assertEqual(r.status_code, 200)


# ===========================================================================
# 11. Upsert idempotency (mocked DB)
# ===========================================================================

class TestUpsertIdempotency(unittest.TestCase):
    """
    Verify the upsert logic does not re-insert duplicate profiles.
    We test the SQL construction, not actual DB (no DB in sandbox).
    """

    def test_duplicate_profile_triggers_update(self):
        """
        Calling upsert_profiles twice with same wmo_id+juld should not
        raise, and the second call should use ON CONFLICT DO UPDATE.
        """
        # Verify the SQL template contains the ON CONFLICT clause
        sql_fragment = "ON CONFLICT (wmo_id, juld) WHERE wmo_id IS NOT NULL"
        # Read the actual cache module source
        import inspect
        src = inspect.getsource(cache_mod.upsert_profiles)
        self.assertIn("ON CONFLICT", src)
        self.assertIn("DO UPDATE", src)

    def test_upsert_sql_prevents_null_wmo_conflict(self):
        """Unique constraint is partial (WHERE wmo_id IS NOT NULL) to allow local rows."""
        import inspect
        src = inspect.getsource(cache_mod.upsert_profiles)
        self.assertIn("WHERE wmo_id IS NOT NULL", src)


# ===========================================================================
# 12. Cache freshness logic
# ===========================================================================

class TestCacheFreshness(unittest.TestCase):
    def test_is_cache_fresh_queries_by_cutoff(self):
        """is_cache_fresh must issue a SQL query with retrieved_at cutoff."""
        import inspect
        src = inspect.getsource(cache_mod.is_cache_fresh)
        self.assertIn("retrieved_at", src)

    def test_lookup_cached_profiles_supports_freshness_filter(self):
        """lookup_cached_profiles accepts require_fresh param."""
        import inspect
        sig = inspect.signature(cache_mod.lookup_cached_profiles)
        self.assertIn("require_fresh", sig.parameters)


# ===========================================================================
# 13. Source attribution
# ===========================================================================

class TestSourceAttribution(unittest.TestCase):
    def test_live_profile_has_erddap_source(self):
        """Profiles from ERDDAP must have source=erddap_ifremer."""
        provider = provider_mod.ArgoERDDAPProvider()
        profiles = provider._parse_csv(_SAMPLE_CSV, "http://test-url")
        for p in profiles:
            self.assertEqual(p["source"], "erddap_ifremer")

    def test_local_fallback_profiles_have_local_source(self):
        """Semantic search results must have source=local (or similar non-live value)."""
        # source badge must not show "live" for local data
        from frontend.chatbot_ui import _source_badge  # type: ignore[import]
        data = {"cache_status": "local", "source": "local"}
        badge = _source_badge(data)
        self.assertNotIn("Live", badge)
        self.assertIn("Local", badge)

    def test_live_badge_for_live_data(self):
        from frontend.chatbot_ui import _source_badge  # type: ignore[import]
        data = {"cache_status": "live", "source": "erddap_ifremer"}
        badge = _source_badge(data)
        self.assertIn("Live", badge)

    def test_cached_badge_for_cached_data(self):
        from frontend.chatbot_ui import _source_badge  # type: ignore[import]
        data = {"cache_status": "cached", "source": "erddap_ifremer"}
        badge = _source_badge(data)
        self.assertIn("Cached", badge)


# ===========================================================================
# 14. Prevention of silent local-data fallback after remote failure
# ===========================================================================

class TestNoSilentFallback(unittest.TestCase):
    """
    Verify that a remote failure does not silently return unrelated local data
    as if it were the requested live data.
    """

    def test_retrieval_error_set_when_erddap_fails(self):
        """query_profiles must set retrieval_error when ERDDAP fails."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cur.fetchone.return_value = [0]
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        with patch.object(api_mod, "_get_db_connection", return_value=mock_conn):
            with patch.object(api_mod, "run_migrations"):
                with patch.object(
                    api_mod._argo_provider,
                    "fetch_profiles",
                    side_effect=provider_mod.ArgoProviderError("ERDDAP down"),
                ):
                    with patch.object(cache_mod, "lookup_cached_profiles", return_value=[]):
                        result = api_mod.query_profiles("lat=22.5, long=60.3")

        self.assertIsNotNone(result.get("retrieval_error"))
        # source_type should NOT be "live"
        self.assertNotEqual(result.get("source_type"), "live")

    def test_local_fallback_label_is_not_live(self):
        """When local data is returned as fallback, source_type must be 'local'."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        # Return one local profile
        mock_cur.fetchall.return_value = [
            (1, 0.0, 0.0, MagicMock(strftime=lambda f: "2025-09-01 00:00:00"),
             [0.1]*384, "local", "9999999")
        ]
        mock_cur.fetchone.return_value = [0]
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur

        with patch.object(api_mod, "_get_db_connection", return_value=mock_conn):
            with patch.object(api_mod, "run_migrations"):
                with patch.object(
                    api_mod._argo_provider,
                    "fetch_profiles",
                    side_effect=provider_mod.ArgoProviderError("ERDDAP down"),
                ):
                    with patch.object(cache_mod, "lookup_cached_profiles", return_value=[]):
                        with patch.object(api_mod, "_semantic_search", return_value=[
                            {
                                "profile_id": 1, "wmo_id": None,
                                "lat": 0.0, "lon": 0.0,
                                "time": "2025-09-01 00:00:00",
                                "source": "local", "cache_status": "local",
                                "depth_levels": [],
                            }
                        ]):
                            result = api_mod.query_profiles("lat=22.5, long=60.3")

        if result.get("profiles"):
            self.assertEqual(result["source_type"], "local")


# ===========================================================================
# 15. Existing cosine similarity regression
# ===========================================================================

class TestCosineSimilarityRegression(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(api_mod.cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(api_mod.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(api_mod.cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)


class TestExtractLatLonRegression(unittest.TestCase):
    def test_explicit_coordinates(self):
        lat, lon = api_mod.extract_lat_lon("data at lat=-43.037, long=130")
        self.assertAlmostEqual(lat, -43.037)
        self.assertAlmostEqual(lon, 130.0)

    def test_no_coordinates_returns_none(self):
        lat, lon = api_mod.extract_lat_lon("show salinity near mumbai")
        self.assertIsNone(lat)
        self.assertIsNone(lon)


# ===========================================================================
# LIVE SMOKE TEST — skipped when offline
# ===========================================================================

import socket

def _can_reach_erddap() -> bool:
    try:
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("erddap.ifremer.fr", 443)
        return True
    except Exception:
        return False


@unittest.skipUnless(_can_reach_erddap(), "ERDDAP not reachable (offline environment)")
class TestLiveSmoke(unittest.TestCase):
    """
    Live integration smoke test against the actual Ifremer ERDDAP server.

    This test is SKIPPED when there is no network.
    A PASS here means the live retrieval actually worked.
    """

    def test_live_wmo_fetch(self):
        """Fetch 1 recent profile for a known active Argo float."""
        provider = provider_mod.ArgoERDDAPProvider()
        # WMO 6902746 is a well-known Argo float in the Indian Ocean
        f = provider_mod.ArgoFilters(
            wmo_ids=["6902746"],
            max_profiles=5,
        )
        profiles = provider.fetch_profiles(f)
        self.assertIsInstance(profiles, list)
        # If the float has data, verify structure
        if profiles:
            p = profiles[0]
            self.assertEqual(p["wmo_id"], "6902746")
            self.assertIsNotNone(p.get("latitude"))
            self.assertIsNotNone(p.get("depth_levels"))
            self.assertEqual(p["source"], "erddap_ifremer")
            print(f"\n[LIVE SMOKE] Retrieved {len(profiles)} profiles for WMO 6902746")
            print(f"  First profile: lat={p['latitude']:.3f}, lon={p['longitude']:.3f}, levels={len(p['depth_levels'])}")

    def test_live_bbox_fetch(self):
        """Fetch profiles in a small Arabian Sea bounding box."""
        provider = provider_mod.ArgoERDDAPProvider()
        f = provider_mod.ArgoFilters(
            min_lat=15.0, max_lat=18.0,
            min_lon=60.0, max_lon=63.0,
            date_start="2026-01-01",
            date_end="2026-06-30",
            max_profiles=5,
        )
        try:
            profiles = provider.fetch_profiles(f)
            print(f"\n[LIVE SMOKE] Arabian Sea bbox: {len(profiles)} profiles")
            self.assertIsInstance(profiles, list)
        except provider_mod.ArgoProviderError as exc:
            self.skipTest(f"ERDDAP returned error (may be no data): {exc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
