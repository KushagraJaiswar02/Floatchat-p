"""Payload-free stage timings and process high-water memory for Render logs."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import logging
import os
from time import perf_counter

logger = logging.getLogger("floatchat.timing")
request_id = ContextVar("request_id", default="background")


def peak_rss_mib():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except ImportError:  # Windows development environment
        return None


@contextmanager
def stage(name):
    started = perf_counter()
    status = "ok"
    logger.info("stage=%s event=start request_id=%s pid=%s", name, request_id.get(), os.getpid())
    try:
        yield
    except BaseException as exc:
        status = type(exc).__name__
        raise
    finally:
        logger.info("stage=%s event=end elapsed_ms=%.1f status=%s request_id=%s pid=%s peak_rss_mib=%s",
                    name, (perf_counter() - started) * 1000, status, request_id.get(), os.getpid(), peak_rss_mib())


def timed(name):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with stage(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate
