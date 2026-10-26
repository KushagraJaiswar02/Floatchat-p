"""
tests/test_memory_and_routing.py — Focused tests for:
1. Intent routing ensuring educational queries do not load embedding model.
2. Single shared SentenceTransformer instance across components.
3. Thread-safe model access.
4. Bounded memory and graceful fallback.
"""

import threading
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch
from api.model_manager import get_shared_sentence_transformer, _model_lock
from api.query_parser import parse_query
import api.query as query_mod
import api.argo_cache as cache_mod
import api.argo_provider as provider_mod
from datetime import datetime, timezone
import requests


CONCEPT_QUERY = "What is the difference between sea surface temperature and temperature at 1000 metres depth?"
OBSERVATION_QUERY = "Find Argo float observations of temperature and salinity in the Indian Ocean."
UI_QUERY = "Open the Map page and try zooming, panning, and hovering over available markers."


class TestProductionIntentRouting(unittest.TestCase):
    def test_classifications(self):
        for text, intent in [
            (CONCEPT_QUERY, "educational"),
            ("What does PSU mean?", "educational"),
            (OBSERVATION_QUERY, "profile_search"),
            ("Hello!", "greeting"),
            (UI_QUERY, "ui_action"),
            ("Hello! Find temperature observations in the Indian Ocean.", "profile_search"),
            ("What is the average temperature at 500 dbar?", "numerical_answer"),
            ("Compare floats 6902746 and 5901234", "comparison"),
        ]:
            with self.subTest(query=text):
                parsed = parse_query(text)
                self.assertEqual(parsed.intent, intent)

    def test_non_data_routes_skip_all_retrieval(self):
        for text, source in [(CONCEPT_QUERY, "educational"), ("Hello!", "system"), (UI_QUERY, "system")]:
            with self.subTest(query=text), ExitStack() as stack:
                guards = [stack.enter_context(patch.object(query_mod, name)) for name in (
                    "_get_db_connection", "_semantic_search", "get_embedding", "lookup_cached_profiles",
                )]
                guards.append(stack.enter_context(patch.object(query_mod._st_model, "_load")))
                guards.append(stack.enter_context(patch.object(query_mod._argo_provider, "fetch_profiles")))
                gemini = stack.enter_context(patch.object(
                    query_mod.gemini_model, "generate_content", side_effect=RuntimeError("offline")
                ))
                response = query_mod.app.test_client().post("/query", json={"query": text})
                self.assertEqual(response.status_code, 200)
                item = response.get_json()[0]
                self.assertEqual(item["source"], source)
                self.assertEqual(item["depth_levels"], [])
                for field in ("profile_id", "lat", "lon", "time", "wmo_id"):
                    self.assertIsNone(item[field])
                for guard in guards:
                    guard.assert_not_called()
                if source == "educational":
                    gemini.assert_called_once()
                    self.assertIn("1000 metres", item["query_explain"])
                    self.assertIn("surface", item["query_explain"])
                else:
                    gemini.assert_not_called()
                if text == UI_QUERY:
                    self.assertIn("can't open pages", item["query_explain"])
                    self.assertIn("haven't performed", item["query_explain"])
                elif text == "Hello!":
                    self.assertIn("Hello!", item["query_explain"])

    def test_observation_request_reaches_structured_live_retrieval(self):
        parsed = parse_query(OBSERVATION_QUERY)
        self.assertTrue(parsed.is_argo_query)
        self.assertEqual(parsed.variables, ["temp", "psal"])
        self.assertEqual(parsed.ambiguities, [])
        self.assertEqual(parsed.filters.region_name, "indian ocean")
        conn = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(query_mod, "_get_db_connection", return_value=conn))
            stack.enter_context(patch.object(query_mod, "run_migrations"))
            cache = stack.enter_context(patch.object(query_mod, "lookup_cached_profiles", return_value=[]))
            fetch = stack.enter_context(patch.object(query_mod._argo_provider, "fetch_profiles", return_value=[]))
            semantic = stack.enter_context(patch.object(query_mod, "_semantic_search"))
            embed = stack.enter_context(patch.object(query_mod, "get_embedding"))
            stack.enter_context(patch.object(query_mod, "ENABLE_LIVE_FETCH", True))
            stack.enter_context(patch.object(query_mod, "ENABLE_LOCAL_FALLBACK", False))
            response = query_mod.app.test_client().post("/query", json={"query": OBSERVATION_QUERY})
            self.assertEqual(response.status_code, 200)
            item = response.get_json()[0]
            self.assertEqual(item["depth_levels"], [])
            self.assertIn("No matching observations", item["query_explain"])
            cache.assert_called_once()
            fetch.assert_called_once()
            filters = fetch.call_args.args[0]
            self.assertEqual(filters.variables, ["temp", "psal"])
            self.assertEqual((filters.min_lat, filters.max_lat, filters.min_lon, filters.max_lon),
                             (-60.0, 30.0, 20.0, 120.0))
            semantic.assert_not_called()
            embed.assert_not_called()
            conn.close.assert_called_once()

    def test_clarification_skips_retrieval(self):
        with patch.object(query_mod, "_get_db_connection") as db:
            response = query_mod.app.test_client().post("/query", json={"query": "Show temperature profiles"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("specify a location", response.get_json()[0]["query_explain"])
            db.assert_not_called()


class TestRetrievalTimeoutRegression(unittest.TestCase):
    def setUp(self):
        self.conn = MagicMock()
        self.raw = {
            "wmo_id": "6902746", "juld": datetime(2026, 10, 26, tzinfo=timezone.utc),
            "latitude": -10.0, "longitude": 70.0, "source": "erddap_ifremer",
            "source_url": "https://example.test/fixture.csvp", "data_mode": "R",
            "depth_levels": [{"pres": 10.0, "temp": 20.0, "psal": 35.0}],
        }  # Unit-test fixture; production always uses the provider's observations.

    def route(self, stack, text=OBSERVATION_QUERY, raw=None):
        stack.enter_context(patch.object(query_mod, "_get_db_connection", return_value=self.conn))
        stack.enter_context(patch.object(query_mod, "run_migrations"))
        stack.enter_context(patch.object(query_mod, "lookup_cached_profiles", return_value=[]))
        stack.enter_context(patch.object(query_mod, "ENABLE_LIVE_FETCH", True))
        stack.enter_context(patch.object(query_mod, "ENABLE_LOCAL_FALLBACK", True))
        self.fetch = stack.enter_context(patch.object(query_mod._argo_provider, "fetch_profiles", return_value=raw or [self.raw]))
        self.write = stack.enter_context(patch.object(query_mod, "upsert_profiles", return_value=(1, 0)))
        self.semantic = stack.enter_context(patch.object(query_mod, "_semantic_search"))
        self.embedding = stack.enter_context(patch.object(cache_mod, "_make_embedding"))
        self.gemini = stack.enter_context(patch.object(query_mod.gemini_model, "generate_content", side_effect=RuntimeError("offline")))
        return query_mod.app.test_client().post("/query", json={"query": text}, headers={"Content-Type": "application/json"})

    def test_broad_query_returns_real_provider_data_without_model(self):
        with ExitStack() as stack:
            response = self.route(stack)
            self.assertEqual(response.status_code, 200)
            item = response.get_json()[0]
            self.assertEqual(item["source"], "erddap_ifremer")
            self.assertEqual(item["wmo_id"], self.raw["wmo_id"])
            self.assertEqual(item["depth_levels"], self.raw["depth_levels"])
            self.assertIn("Recent sample:", item["query_explain"])
            f = self.fetch.call_args.args[0]
            self.assertIsNotNone(f.date_start)
            self.assertIsNotNone(f.date_end)
            self.assertEqual(f.max_profiles, query_mod.TOP_K)
            self.write.assert_called_once_with(self.conn, [self.raw], skip_embedding=True)
            self.embedding.assert_not_called()
            self.semantic.assert_not_called()
            self.assertEqual(self.gemini.call_args.kwargs["request_options"], {"timeout": 4, "retry": None})
            self.assertTrue(response.headers["X-Request-ID"])

    def test_explicit_dates_are_preserved(self):
        with ExitStack() as stack:
            self.route(stack, OBSERVATION_QUERY + " between 2020-01-01 and 2020-02-01")
            f = self.fetch.call_args.args[0]
            self.assertEqual((f.date_start, f.date_end), ("2020-01-01", "2020-02-01"))

    def test_timeout_returns_frontend_error_without_semantic_model(self):
        with ExitStack() as stack:
            self.route(stack)  # Set up the same route, then reproduce its upstream failure.
            self.fetch.side_effect = provider_mod.ArgoProviderError("ERDDAP timed out")
            self.semantic.reset_mock()
            self.gemini.reset_mock()
            response = query_mod.app.test_client().post("/query", json={"query": OBSERVATION_QUERY})
            self.assertEqual(response.status_code, 200)
            item = response.get_json()[0]
            self.assertEqual(item["source"], "error")
            self.assertEqual(item["depth_levels"], [])
            self.assertIn("ERDDAP timed out", item["query_explain"])
            self.semantic.assert_not_called()
            self.gemini.assert_not_called()

    def test_cache_write_failure_still_serves_observations(self):
        with ExitStack() as stack:
            self.route(stack)
            self.write.side_effect = RuntimeError("cache unavailable")
            response = query_mod.app.test_client().post("/query", json={"query": OBSERVATION_QUERY})
            self.assertEqual(response.get_json()[0]["wmo_id"], self.raw["wmo_id"])
            self.conn.rollback.assert_called_once()

    def test_skip_embedding_is_valid_for_not_null_schema(self):
        self.conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (42, True)
        with patch.object(cache_mod, "_make_embedding") as embed, patch.object(cache_mod.psycopg2.extras, "execute_batch"):
            cache_mod.upsert_profiles(self.conn, [self.raw], skip_embedding=True)
            embed.assert_not_called()
            call = self.conn.cursor.return_value.__enter__.return_value.execute.call_args
            self.assertEqual(call.args[1][-2], [])
            self.assertIn("ELSE profiles.embedding", call.args[0])

    def test_upstream_timeout_is_bounded_and_not_retried(self):
        for exc in (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
            with self.subTest(error=exc.__name__), patch("requests.get", side_effect=exc) as get:
                with self.assertRaises(provider_mod.ArgoProviderError):
                    provider_mod.ArgoERDDAPProvider().fetch_profiles(provider_mod.ArgoFilters(region_name="indian ocean"))
                get.assert_called_once()
                self.assertEqual(get.call_args.kwargs["timeout"][0], 3)
                self.assertLessEqual(get.call_args.kwargs["timeout"][1], 12)
                self.assertTrue(get.call_args.kwargs["stream"])
                self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_download_byte_limit_closes_response(self):
        response = MagicMock(status_code=200)
        response.raw.read1.side_effect = [b"x" * 11, b""]
        with patch("requests.get", return_value=response), patch.object(provider_mod, "MAX_RESPONSE_BYTES", 10):
            with self.assertRaisesRegex(provider_mod.ArgoProviderError, "safe size limit"):
                provider_mod.ArgoERDDAPProvider()._request_csv("https://example.test/fixture")
        response.close.assert_called_once()

    def test_download_deadline_closes_response(self):
        response = MagicMock(status_code=200)
        response.raw.read1.side_effect = [b"x", b""]
        with patch("requests.get", return_value=response), patch.object(provider_mod, "monotonic", side_effect=[0, 60]):
            with self.assertRaisesRegex(provider_mod.ArgoProviderError, "timed out"):
                provider_mod.ArgoERDDAPProvider()._request_csv("https://example.test/fixture")
        response.close.assert_called_once()

    def test_partial_download_error_is_not_silently_empty(self):
        response = MagicMock(status_code=200)
        response.raw.read1.side_effect = requests.exceptions.ChunkedEncodingError("interrupted")
        with patch("requests.get", return_value=response):
            with self.assertRaises(provider_mod.ArgoProviderError):
                provider_mod.ArgoERDDAPProvider()._request_csv("https://example.test/fixture")
        response.close.assert_called_once()

    def test_multi_wmo_failure_does_not_fan_out_retries(self):
        response = MagicMock(status_code=503)
        with patch("requests.get", return_value=response) as get:
            with self.assertRaises(provider_mod.ArgoProviderError):
                provider_mod.ArgoERDDAPProvider().fetch_multiple_wmo(provider_mod.ArgoFilters(wmo_ids=["6902746", "5901234"]))
            get.assert_called_once()
        response.close.assert_called_once()

    def test_timing_logs_do_not_contain_query(self):
        with self.assertLogs("floatchat.timing", level="INFO") as logs:
            with ExitStack() as stack:
                self.route(stack)
        output = " ".join(logs.output)
        self.assertNotIn(OBSERVATION_QUERY, output)
        for stage_name in ("intent_classification", "response_format", "response_serialization", "request_total"):
            self.assertIn("stage=" + stage_name, output)
        self.assertIn("elapsed_ms=", output)
        self.assertIn("peak_rss_mib=", output)


class TestModelManagerAndRouting(unittest.TestCase):

    def test_shared_model_instance(self):
        """Verify argo_cache and query use the exact same model instance."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [0.1] * 384

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance):
            with patch("api.model_manager._shared_st_instance", None):
                m1 = cache_mod._get_embedding_model()
                m2 = query_mod._st_model._load()
                self.assertIs(m1, m2, "Model instance must be a singleton")

    def test_educational_query_does_not_load_model(self):
        """Verify educational queries never touch the embedding model."""
        client = query_mod.app.test_client()
        with patch.object(query_mod.gemini_model, "generate_content", return_value=MagicMock(text="Salinity is dissolved salt concentration.")), \
                patch.object(query_mod, "get_embedding") as mock_get_emb:
            r = client.post(
                "/query",
                json={"query": "what is salinity in oceanography?"},
            )
            self.assertEqual(r.status_code, 200)
            mock_get_emb.assert_not_called()

    def test_concurrent_model_access_thread_safety(self):
        """Verify concurrent requests serialize through the thread lock without crashing."""
        mock_instance = MagicMock()
        mock_instance.encode.return_value = [0.2] * 384
        results = []

        def worker():
            with patch("sentence_transformers.SentenceTransformer", return_value=mock_instance):
                m = get_shared_sentence_transformer()
                results.append(m)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 5)
        for res in results:
            self.assertIs(res, results[0])

    def test_semantic_search_empty_table_skips_embedding_model(self):
        """Verify semantic search checks DB first and skips encoding if no rows exist."""
        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur
        mock_cur.fetchall.return_value = []  # No embeddings in DB

        with patch.object(query_mod, "get_embedding") as mock_get_emb:
            res = query_mod._semantic_search(mock_conn, "temperature near equator")
            self.assertEqual(res, [])
            mock_get_emb.assert_not_called()


if __name__ == "__main__":
    unittest.main()
