"""
run_server.py — Production entry point for FloatChat API using Waitress.
"""

import logging
import os
import sys

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("floatchat.server")


def main():
    # Enforce CPU & single-thread execution for PyTorch to minimize memory footprint
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass

    port_str = os.environ.get("PORT", "10000")
    try:
        port = int(port_str)
    except ValueError:
        logger.warning("Invalid PORT environment variable %r, defaulting to 10000", port_str)
        port = 10000

    host = "0.0.0.0"
    logger.info("Initializing FloatChat backend server (low-memory CPU configuration)...")

    from api.query import app
    from waitress import serve

    threads = int(os.environ.get("WAITRESS_THREADS", "4"))
    logger.info("Serving FloatChat Flask API on http://%s:%d (threads: %d)", host, port, threads)

    serve(app, host=host, port=port, threads=threads)


if __name__ == "__main__":
    main()
