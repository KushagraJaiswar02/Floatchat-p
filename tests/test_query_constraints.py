"""Deterministic constraint enforcement; all observations here are test fixtures."""
from contextlib import ExitStack
from datetime import date, datetime, timedelta, timezone
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

import requests
import api.query as query
import api.argo_cache as cache
from api.argo_provider import ArgoERDDAPProvider, ArgoFilters, ArgoProviderError, ProfileBatch
from api.query_contract import depth_from_pressure, filter_depth_levels, pressure_at_depth
from api.query_parser import parse_query
from frontend.chatbot_ui import is_ocean_data_query

REFERENCE = date(2026, 10, 26)
QUESTION = "Find temperature and salinity observations between 0 and 1000 metres depth. lat 70 long 40 last month"
HEADER = "platform_number,time,latitude,longitude,pres,temp,psal\n"


def response(text, status=200):
    result = MagicMock(status_code=status)
    result.raw.read1.side_effect = [text.encode(), b""]
    return result


class TestCoordinateAndDateContract(unittest.TestCase):
    def test_coordinate_forms_share_frontend_backend_contract(self):
        for text in ("lat 70 long 40", "latitude 70 longitude 40", "latitude: 70, longitude: 40",
                     "lat=70, lon=40", "70°N, 40°E"):
            with self.subTest(text=text):
                self.assertTrue(is_ocean_data_query(text))
                parsed = parse_query(text)
                self.assertEqual(parsed.ambiguities, [])
                self.assertEqual((parsed.filters.min_lat, parsed.filters.max_lat,
                                  parsed.filters.min_lon, parsed.filters.max_lon), (69, 71, 39, 41))

    def test_invalid_coordinates_reach_validation_without_retrieval(self):
        for text in ("lat 91 long 40", "lat -91 long 40", "lat 70 long 181", "lat 70 long -181"):
            with self.subTest(text=text), patch.object(query, "_get_db_connection") as db:
                self.assertTrue(is_ocean_data_query(text))
                self.assertTrue(parse_query(text).ambiguities)
                result = query.app.test_client().post("/query", json={"query": text}).get_json()[0]
                self.assertEqual(result["retrieval_status"], "invalid_filters")
                self.assertNotIn("No matching observations", result["query_explain"])
                db.assert_not_called()

    def test_coordinate_signs_hemispheres_and_limits(self):
        self.assertFalse(is_ocean_data_query("How long should I cook pasta?"))
        for text in ("lat -70 long -40", "lat 70 S long 40 W", "70 S, 40 W"):
            with self.subTest(text=text):
                f = parse_query(text).filters
                self.assertEqual((f.min_lat, f.max_lat, f.min_lon, f.max_lon), (-71, -69, -41, -39))
        for text in ("lat 90 long 180", "lat -90 long -180"):
            self.assertEqual(parse_query(text).ambiguities, [])
        self.assertTrue(parse_query("lat -70 N long 40 E").ambiguities)

    def test_calendar_relative_dates(self):
        for phrase, expected in [
            ("last month", ("2026-09-01", "2026-09-30")),
            ("this month", ("2026-10-01", "2026-10-31")),
            ("last week", ("2026-10-19", "2026-10-25")),
        ]:
            with self.subTest(phrase=phrase):
                f = parse_query("lat 70 long 40 " + phrase, reference_date=REFERENCE).filters
                self.assertEqual((f.date_start, f.date_end), expected)

    def test_year_leap_and_utc_boundaries(self):
        for ref, expected in [(date(2026, 1, 1), ("2025-12-01", "2025-12-31")),
                              (date(2024, 3, 1), ("2024-02-01", "2024-02-29"))]:
            f = parse_query("temperature last month", reference_date=ref).filters
            self.assertEqual((f.date_start, f.date_end), expected)
        ref = datetime(2026, 10, 1, 0, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
        f = parse_query("temperature this month", reference_date=ref).filters
        self.assertEqual((f.date_start, f.date_end), ("2026-09-01", "2026-09-30"))

    def test_explicit_dates_win_and_invalid_dates_do_not_default(self):
        f = parse_query("lat 70 long 40 last month between 2024-02-01 and 2024-02-29", reference_date=REFERENCE).filters
        self.assertEqual((f.date_start, f.date_end), ("2024-02-01", "2024-02-29"))
        for text in ("lat 70 long 40 on 2026-02-31", "lat 70 long 40 last 30 days",
                     "lat 70 long 40 between 2026-10-01 and 2026-09-01"):
            self.assertTrue(parse_query(text, reference_date=REFERENCE).ambiguities)
        # The existing explicit month parser must handle February too.
        f = parse_query("temperature February 2024").filters
        self.assertEqual((f.date_start, f.date_end), ("2024-02-01", "2024-02-29"))


class TestDepthRetrieval(unittest.TestCase):
    def test_unesco_reference_and_inclusive_boundaries(self):
        # Published UNESCO 1983 check value, independent of our inverse helper.
        self.assertAlmostEqual(depth_from_pressure(10000, 30), 9712.653, places=3)
        levels = [{"pres": p, "temp": t, "psal": s} for p, t, s in
                  [(0, 0.0, 0.0), (500, 7.0, None), (pressure_at_depth(1000, 70), None, 35.0),
                   (pressure_at_depth(1000.01, 70), 1.0, 36.0), (None, 9, 33), (-1, 8, 34)]]
        kept = filter_depth_levels(levels, 70, 0, 1000)
        self.assertEqual(len(kept), 3)
        self.assertEqual([i["temp"] for i in kept], [0.0, 7.0, None])
        self.assertEqual([i["psal"] for i in kept], [0.0, None, 35.0])
        self.assertTrue(all(0 <= i["depth_m"] <= 1000 for i in kept))
        self.assertEqual(kept[0]["depth_m"], 0)

    def test_provider_propagates_depth_and_filters_real_rows(self):
        f = parse_query(QUESTION, reference_date=REFERENCE).filters
        csv = HEADER + "6902746,2026-09-12T00:00:00Z,70,40,0,0,0\n" + \
            "6902746,2026-09-12T00:00:00Z,70,40,500,7,35\n" + \
            "6902746,2026-09-12T00:00:00Z,70,40,1500,2,34\n"
        with patch("requests.get", return_value=response(csv)) as get:
            profiles = ArgoERDDAPProvider().fetch_profiles(f)
        url = get.call_args.args[0]
        for constraint in ('latitude>=69.0', 'longitude<=41.0', 'time>="2026-09-01T00:00:00Z"',
                           'time<"2026-10-01T00:00:00Z"', 'pres>=0', 'pres<='):
            self.assertIn(constraint, url)
        self.assertEqual([v["pres"] for v in profiles[0]["depth_levels"]], [0, 500])
        self.assertTrue(profiles.complete)

    def test_multi_float_depth_constraints_are_not_dropped(self):
        f = ArgoFilters(wmo_ids=["6902746", "5901234"], min_depth_m=0, max_depth_m=1000)
        with patch.object(ArgoERDDAPProvider, "_request_csv", return_value=HEADER) as get:
            ArgoERDDAPProvider().fetch_multiple_wmo(f)
        self.assertIn("pres<=", get.call_args.args[0])

    def test_invalid_depth_range(self):
        for bounds in ("-1 and 1000", "1000 and 0", "0 and 13000"):
            self.assertTrue(parse_query("Find temperature lat 70 long 40 between " + bounds + " metres depth").ambiguities)
        self.assertTrue(parse_query("Find temperature lat 70 long 40 between 0 and 1000 metres depth exclusive").ambiguities)

    def test_cache_filters_before_profile_and_level_limits(self):
        # Execute the actual cache SQL on an isolated in-memory database.
        db = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
        self.addCleanup(db.close)
        db.executescript("""
            CREATE TABLE profiles (id INTEGER, wmo_id TEXT, juld TIMESTAMP, latitude REAL,
                longitude REAL, source TEXT, source_url TEXT, retrieved_at TIMESTAMP, data_mode TEXT);
            CREATE TABLE profile_levels (profile_id INTEGER, n_levels INTEGER, pres REAL,
                temp REAL, psal REAL, pres_qc TEXT, temp_qc TEXT, psal_qc TEXT);
        """)
        db.executemany("INSERT INTO profiles VALUES (?,?,?,?,?,?,?,?,?)", [
            (1, "6902746", datetime(2026, 9, 19, 23, 59, 59, 999999), 70, 40, "local", None, None, "R"),
            (2, "5901234", datetime(2026, 9, 19, 23, 59, 59, 999998), 70, 40, "local", None, None, "R"),
        ])
        db.executemany("INSERT INTO profile_levels VALUES (?,?,?,?,?,?,?,?)",
                       [(1, 0, 1500, 3, 35, "1", "1", "1")] +
                       [(2, i, 1500, 3, 35, "1", "1", "1") for i in range(250)] +
                       [(2, 250, 0, 0, 0, "1", "1", "1"), (2, 251, 500, 7, 35, "1", "1", "1")])

        class Cursor:
            def __enter__(self):
                self.cur = db.cursor()
                return self
            def __exit__(self, *_):
                self.cur.close()
            def execute(self, sql, params):
                self.cur.execute(sql.replace("%s", "?"), params)
            def fetchall(self):
                return self.cur.fetchall()

        conn = MagicMock()
        conn.cursor.side_effect = Cursor
        result = cache.lookup_cached_profiles(conn, min_lat=69, max_lat=71, min_lon=39, max_lon=41,
                                              date_start="2026-09-01", date_end="2026-09-19",
                                              min_depth_m=0, max_depth_m=1000, max_results=1)
        self.assertEqual([p["profile_id"] for p in result], [2])
        self.assertEqual([v["pres"] for v in result[0]["depth_levels"]], [0, 500])
        self.assertEqual(result[0]["depth_levels"][0]["temp"], 0)
        self.assertEqual(result[0]["depth_levels"][0]["psal"], 0)


class TestConstraintRouteAndEmptyResults(unittest.TestCase):
    def route_context(self, stack):
        self.conn = MagicMock()
        stack.enter_context(patch.object(query, "_get_db_connection", return_value=self.conn))
        stack.enter_context(patch.object(query, "run_migrations"))
        self.lookup = stack.enter_context(patch.object(query, "lookup_cached_profiles", return_value=[]))
        self.write = stack.enter_context(patch.object(query, "upsert_profiles"))
        self.semantic = stack.enter_context(patch.object(query, "_semantic_search"))
        stack.enter_context(patch.object(query, "ENABLE_LIVE_FETCH", True))
        self.gemini = stack.enter_context(patch.object(query.gemini_model, "generate_content", return_value=MagicMock(text="Fixture explanation")))
        stack.enter_context(patch.object(query, "parse_query", side_effect=lambda text, **_: parse_query(text, reference_date=REFERENCE)))

    def test_end_to_end_frontend_parse_provider_response(self):
        csv = HEADER + "6902746,2026-09-12T00:00:00Z,70,40,0,0,0\n" + \
            "6902746,2026-09-12T00:00:00Z,70,40,500,7,35\n" + \
            "6902746,2026-09-12T00:00:00Z,70,40,1500,2,35\n"
        self.assertTrue(is_ocean_data_query(QUESTION))
        with ExitStack() as stack:
            self.route_context(stack)
            get = stack.enter_context(patch("requests.get", return_value=response(csv)))
            r = query.app.test_client().post("/query", json={"query": QUESTION})
            item = r.get_json()[0]
            self.assertEqual(r.status_code, 200)
            f = item["filters_used"]
            self.assertEqual((f["date_start"], f["date_end"]), ("2026-09-01", "2026-09-30"))
            self.assertEqual((f["min_depth_m"], f["max_depth_m"]), (0, 1000))
            self.assertEqual((f["min_lat"], f["max_lat"], f["min_lon"], f["max_lon"]), (69, 71, 39, 41))
            self.assertEqual(item["retrieval_status"], "success")
            self.assertEqual([i["pres"] for i in item["depth_levels"]], [0, 500])
            self.assertTrue(all(0 <= i["depth_m"] <= 1000 for i in item["depth_levels"]))
            self.assertEqual(item["depth_levels"][0]["temp"], 0)
            self.assertEqual(self.lookup.call_args.kwargs["max_depth_m"], 1000)
            self.assertEqual(self.lookup.call_args.kwargs["date_end"], "2026-09-30")
            self.assertIn('time>="2026-09-01', get.call_args.args[0])
            self.assertIn("UNESCO 1983", item["query_explain"])
            self.write.assert_not_called()  # Never overwrite complete cache rows with a depth subset.
            self.semantic.assert_not_called()

    def test_confirmed_zero_failure_and_incomplete_are_distinct(self):
        for batch, error, expected in [(ProfileBatch(), None, "empty"),
                                        (None, ArgoProviderError("ERDDAP timed out"), "failed"),
                                        (ProfileBatch(complete=False), None, "incomplete")]:
            with self.subTest(status=expected), ExitStack() as stack:
                self.route_context(stack)
                stack.enter_context(patch.object(query._argo_provider, "fetch_profiles", return_value=batch, side_effect=error))
                item = query.app.test_client().post("/query", json={"query": QUESTION}).get_json()[0]
                self.assertEqual(item["retrieval_status"], expected)
                self.assertEqual("No matching observations" in item["query_explain"], expected == "empty")
                self.assertIn("2026-09-01", item["query_explain"])
                self.assertIn("latitude [69.0, 71.0]", item["query_explain"])
                self.assertIn("depth [0.0, 1000.0]", item["query_explain"])
                self.assertEqual(item["depth_levels"], [])
                self.semantic.assert_not_called()
                self.gemini.assert_not_called()

    def test_no_live_retrieval_does_not_establish_zero(self):
        with ExitStack() as stack:
            self.route_context(stack)
            stack.enter_context(patch.object(query, "ENABLE_LIVE_FETCH", False))
            result = query.query_profiles(QUESTION, reference_date=REFERENCE)
            self.assertEqual(result["retrieval_status"], "incomplete")

    def test_unknown_404_and_empty_200_are_errors(self):
        for text, status in [("Unknown dataset", 404), ("", 404), ("", 200)]:
            with self.subTest(status=status, text=text), patch("requests.get", return_value=response(text, status)):
                with self.assertRaises(ArgoProviderError):
                    ArgoERDDAPProvider().fetch_profiles(ArgoFilters(wmo_ids=["6902746"]))

    def test_safety_cap_or_unknown_depth_cannot_prove_zero(self):
        for csv, max_profiles in [
            (HEADER + "6902746,2026-09-12T00:00:00Z,70,40,1500,2,35\n" * 300, 1),
            (HEADER + "6902746,2026-09-12T00:00:00Z,70,40,,2,35\n", 20),
        ]:
            with patch.object(ArgoERDDAPProvider, "_request_csv", return_value=csv):
                result = ArgoERDDAPProvider().fetch_profiles(ArgoFilters(wmo_ids=["6902746"], min_depth_m=0, max_depth_m=1000, max_profiles=max_profiles))
            self.assertEqual(result, [])
            self.assertFalse(result.complete)


if __name__ == "__main__":
    unittest.main()
