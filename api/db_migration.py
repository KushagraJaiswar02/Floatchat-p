"""
api/db_migration.py — Additive schema migration for FloatChat live Argo integration.

Adds provenance/source columns to existing tables and creates a unique constraint
on (wmo_id, juld) for idempotent upserts. Safe to run multiple times (idempotent).

Run once before starting the live integration:
    python api/db_migration.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger("floatchat.migration")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

DB_CONFIG = {
    "host": os.getenv("HOST_NAME", "localhost"),
    "database": os.getenv("DATABASE_NAME", "floatchatai"),
    "user": os.getenv("USER_NAME"),
    "password": os.getenv("PASSWORD"),
    "port": int(os.getenv("PORT", "5432")),
}

# ---------------------------------------------------------------------------
# Migration steps — each is idempotent
# ---------------------------------------------------------------------------

_MIGRATIONS: list[tuple[str, str]] = [
    # Step 1: Add wmo_id to profiles
    (
        "add_wmo_id_to_profiles",
        """
        ALTER TABLE profiles
          ADD COLUMN IF NOT EXISTS wmo_id VARCHAR(20) DEFAULT NULL;
        """,
    ),
    # Step 2: Add source columns to profiles
    (
        "add_source_columns_to_profiles",
        """
        ALTER TABLE profiles
          ADD COLUMN IF NOT EXISTS source VARCHAR(50) DEFAULT 'local',
          ADD COLUMN IF NOT EXISTS source_url TEXT DEFAULT NULL,
          ADD COLUMN IF NOT EXISTS retrieved_at TIMESTAMPTZ DEFAULT NULL,
          ADD COLUMN IF NOT EXISTS data_mode CHAR(1) DEFAULT NULL;
        """,
    ),
    # Step 3: Add QC columns to profile_levels
    (
        "add_qc_columns_to_profile_levels",
        """
        ALTER TABLE profile_levels
          ADD COLUMN IF NOT EXISTS pres_qc CHAR(1) DEFAULT NULL,
          ADD COLUMN IF NOT EXISTS temp_qc CHAR(1) DEFAULT NULL,
          ADD COLUMN IF NOT EXISTS psal_qc CHAR(1) DEFAULT NULL;
        """,
    ),
    # Step 4: Index wmo_id for fast lookup
    (
        "index_profiles_wmo_id",
        """
        CREATE INDEX IF NOT EXISTS idx_profiles_wmo_id
          ON profiles (wmo_id);
        """,
    ),
    # Step 5: Unique constraint for idempotent upserts (wmo_id, juld)
    # Only applies to rows that have wmo_id (live data); local rows have NULL wmo_id
    (
        "unique_profiles_wmo_juld",
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uniq_profiles_wmo_juld
          ON profiles (wmo_id, juld)
          WHERE wmo_id IS NOT NULL;
        """,
    ),
    # Step 6: Index on source for filtering
    (
        "index_profiles_source",
        """
        CREATE INDEX IF NOT EXISTS idx_profiles_source
          ON profiles (source);
        """,
    ),
    # Step 7: Mark existing local rows
    (
        "mark_local_rows",
        """
        UPDATE profiles SET source = 'local'
          WHERE source IS NULL;
        """,
    ),
]


def run_migrations(conn: "psycopg2.extensions.connection") -> None:
    """Apply all pending migrations in order."""
    with conn.cursor() as cur:
        # Ensure migration tracking table exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                name VARCHAR(100) PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )
        conn.commit()

        for name, sql in _MIGRATIONS:
            cur.execute(
                "SELECT 1 FROM schema_migrations WHERE name = %s", (name,)
            )
            if cur.fetchone():
                logger.info("Migration already applied: %s", name)
                continue
            logger.info("Applying migration: %s", name)
            cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (name) VALUES (%s)", (name,)
            )
            conn.commit()
            logger.info("Migration applied: %s", name)


if __name__ == "__main__":
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        run_migrations(conn)
        print("All migrations applied successfully.")
    finally:
        conn.close()
