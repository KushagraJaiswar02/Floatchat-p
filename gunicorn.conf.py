"""
gunicorn.conf.py — Gunicorn configuration for FloatChat API service.
"""

import os

# Server socket
bind = f"0.0.0.0:{os.getenv('PORT', '10000')}"

# Worker processes
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 2

# Logging
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "debug")

# Crucial for Render / container health checks:
# preload_app = False ensures Gunicorn binds the listening port before
# loading application workers.
preload_app = False
