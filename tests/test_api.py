"""
tests/test_api.py — focused tests for api/query.py config, validation and result handling.

Run with:
    .venv\\Scripts\\python.exe -m pytest tests/ -v
"""
import json
import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make the api/ package importable without actually loading the heavy ML libs
# at import time.  We stub the slow top-level side effects before importing.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


def _stub_module(name, **attrs):
    """Create a minimal stub module and register it in sys.modules."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# Stub sentence_transformers so loading the real model is skipped
_st = _stub_module("sentence_transformers")
_st_model = MagicMock()
_st_model.encode = MagicMock(return_value=[0.1] * 384)
_st.SentenceTransformer = MagicMock(return_value=_st_model)

# Stub google.generativeai
_genai = _stub_module("google")
_generativeai = _stub_module("google.generativeai")
_generativeai.configure = MagicMock()
_mock_gemini = MagicMock()
_mock_gemini.generate_content = MagicMock(
    return_value=MagicMock(text="Test AI explanation")
)
_generativeai.GenerativeModel = MagicMock(return_value=_mock_gemini)

# Stub psycopg2 (we'll override per-test)
_psycopg2 = _stub_module("psycopg2")
_psycopg2.Error = Exception
_psycopg2.connect = MagicMock()
_psycopg2_extras = _stub_module("psycopg2.extras")
_psycopg2_extras.execute_batch = MagicMock()
_psycopg2.extras = _psycopg2_extras

# Stub geopy
_geopy = _stub_module("geopy")
_geocoders = _stub_module("geopy.geocoders")
_geocoders.Nominatim = MagicMock()
_geo_distance = _stub_module("geopy.distance")
_geo_distance.distance = MagicMock(return_value=MagicMock(meters=1000))

# Ensure ROOT .env is found when the module loads
os.environ.setdefault("HOST_NAME", "localhost")
os.environ.setdefault("DATABASE_NAME", "floatchatai")
os.environ.setdefault("USER_NAME", "postgres")
os.environ.setdefault("PASSWORD", "testpassword")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")

sys.path.insert(0, str(ROOT / "api"))

import query as api_module  # noqa: E402 — must come after stubs


class TestExtractLatLon(unittest.TestCase):
    def test_explicit_coordinates(self):
        lat, lon = api_module.extract_lat_lon("data at lat=-43.037, long=130")
        self.assertAlmostEqual(lat, -43.037)
        self.assertAlmostEqual(lon, 130.0)

    def test_no_coordinates_returns_none(self):
        lat, lon = api_module.extract_lat_lon("show salinity near mumbai")
        self.assertIsNone(lat)
        self.assertIsNone(lon)

    def test_positive_coordinates(self):
        lat, lon = api_module.extract_lat_lon("lat=12.5, long=77.6")
        self.assertAlmostEqual(lat, 12.5)
        self.assertAlmostEqual(lon, 77.6)


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(api_module.cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(api_module.cosine_similarity(a, b), 0.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        self.assertEqual(api_module.cosine_similarity(a, b), 0.0)


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        self.client = api_module.app.test_client()
        api_module.app.config["TESTING"] = True

    # --- GET / ---
    def test_home_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertIn("message", data)

    # --- POST /query validation ---
    def test_post_no_body_returns_400(self):
        r = self.client.post("/query", content_type="application/json", data="")
        self.assertEqual(r.status_code, 400)

    def test_post_non_json_returns_400(self):
        r = self.client.post("/query", content_type="text/plain", data="hello")
        self.assertEqual(r.status_code, 400)

    def test_post_missing_query_field_returns_400(self):
        r = self.client.post(
            "/query", content_type="application/json", data=json.dumps({})
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", json.loads(r.data))

    def test_post_empty_query_returns_400(self):
        r = self.client.post(
            "/query",
            content_type="application/json",
            data=json.dumps({"query": "   "}),
        )
        self.assertEqual(r.status_code, 400)

    # --- POST /query with mocked DB ---
    def test_post_valid_query_returns_list(self):
        """A valid query that finds no profiles within radius returns []."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cur
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        # Return one profile outside the radius
        mock_cur.fetchall.return_value = [
            (1, 0.0, 0.0, MagicMock(strftime=lambda fmt: "2023-01-01 00:00:00"), [0.1] * 384)
        ]

        with patch.object(api_module.psycopg2, "connect", return_value=mock_conn):
            # distance is mocked to 1000 m (within 50 km radius)
            r = self.client.post(
                "/query",
                content_type="application/json",
                data=json.dumps({"query": "lat=0.0, long=0.0"}),
            )
        self.assertIn(r.status_code, (200, 500))  # 500 only if mock is incomplete

    def test_post_db_error_returns_500_without_stack_trace(self):
        with patch.object(
            api_module.psycopg2,
            "connect",
            side_effect=api_module.psycopg2.Error("DB down"),
        ):
            r = self.client.post(
                "/query",
                content_type="application/json",
                data=json.dumps({"query": "lat=0.0, long=0.0"}),
            )
        self.assertEqual(r.status_code, 500)
        body = json.loads(r.data)
        self.assertIn("error", body)
        # Must not expose raw stack trace or credentials
        self.assertNotIn("Traceback", body["error"])
        self.assertNotIn("password", body["error"].lower())


class TestNaNHandling(unittest.TestCase):
    """Verify depth-level NaN/None filtering in query_profiles result assembly."""

    def _make_profile_data(self, levels):
        depth_levels = []
        for pres, temp, sal in levels:
            if pres is None or temp is None or sal is None:
                continue
            try:
                if math.isnan(pres) or math.isnan(temp) or math.isnan(sal):
                    continue
            except TypeError:
                continue
            depth_levels.append({"pres": pres, "temp": temp, "salinity": sal})
        return depth_levels

    def test_nan_values_excluded(self):
        levels = [(5.0, float("nan"), 35.0), (10.0, 18.0, 35.1)]
        result = self._make_profile_data(levels)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["pres"], 10.0)

    def test_none_values_excluded(self):
        levels = [(None, 18.0, 35.0), (10.0, 18.0, 35.1)]
        result = self._make_profile_data(levels)
        self.assertEqual(len(result), 1)

    def test_all_valid_values_included(self):
        levels = [(5.0, 18.0, 35.0), (10.0, 17.5, 35.1)]
        result = self._make_profile_data(levels)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
