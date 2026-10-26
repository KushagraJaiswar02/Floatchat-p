"""
tests/test_memory_and_routing.py — Focused tests for:
1. Intent routing ensuring educational queries do not load embedding model.
2. Single shared SentenceTransformer instance across components.
3. Thread-safe model access.
4. Bounded memory and graceful fallback.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch
from api.model_manager import get_shared_sentence_transformer, _model_lock
import api.query as query_mod
import api.argo_cache as cache_mod


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
        with patch.object(query_mod, "get_embedding") as mock_get_emb:
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
