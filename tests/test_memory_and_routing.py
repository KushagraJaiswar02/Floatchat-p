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
            self.assertEqual(response.get_json(), [])  # Empty provider results must stay empty.
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
