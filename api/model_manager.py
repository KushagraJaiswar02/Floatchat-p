"""
api/model_manager.py — Centralized, thread-safe, single-instance SentenceTransformer loader.

Guarantees:
1. Exactly one SentenceTransformer instance across the entire application (argo_cache, query, etc.).
2. Thread-safe singleton creation (double-checked locking).
3. PyTorch constrained to CPU, single thread, and no CUDA caching.
4. Fast GC and memory safety for memory-constrained environments (e.g., Render 512 MiB).
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger("floatchat.models")

_model_lock = threading.Lock()
_shared_st_instance: Optional[Any] = None


def get_shared_sentence_transformer(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Return the shared SentenceTransformer singleton, creating it safely if needed."""
    global _shared_st_instance
    if _shared_st_instance is not None:
        return _shared_st_instance

    with _model_lock:
        if _shared_st_instance is not None:
            return _shared_st_instance

        logger.info("Initializing singleton SentenceTransformer(%s) for low-memory CPU...", model_name)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

        try:
            import torch
            torch.set_num_threads(1)
            if hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(1)
        except Exception:
            pass

        from sentence_transformers import SentenceTransformer

        # device="cpu" ensures no CUDA context allocation
        instance = SentenceTransformer(model_name, device="cpu")
        instance.eval()  # ensure evaluation mode

        # Run garbage collection right after loading weights into memory
        gc.collect()

        logger.info("Shared SentenceTransformer loaded and ready.")
        _shared_st_instance = instance
        return _shared_st_instance
