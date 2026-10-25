
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import psycopg2
from psycopg2.extras import execute_batch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("HOST_NAME", "localhost"),
    "database": os.getenv("DATABASE_NAME", "floatchatai"),
    "user": os.getenv("USER_NAME"),
    "password": os.getenv("PASSWORD"),
    "port": int(os.getenv("PORT", "5432")),
}

DATA_DIR = ROOT_DIR / "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_number(value):
    """Convert a numeric value to a PostgreSQL-safe float or NULL."""
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def get_timestamp(value):
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def main():
    required = ("user", "password")
    missing = [key for key in required if not DB_CONFIG[key]]
    if missing:
        raise RuntimeError(
            "Missing database settings in .env: " + ", ".join(missing)
        )

    files = sorted(DATA_DIR.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {DATA_DIR}")

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM profiles")
            existing = cur.fetchone()[0]

        if existing:
            raise RuntimeError(
                f"Database already contains {existing} profiles. "
                "Stopping to avoid duplicate imports."
            )

        for file_path in files:
            print(f"Reading {file_path.name}")

            with xr.open_dataset(file_path) as ds:
                for prof in range(ds.sizes["N_PROF"]):
                    juld = get_timestamp(ds["JULD"].values[prof])
                    lat = get_number(ds["LATITUDE"].values[prof])
                    lon = get_number(ds["LONGITUDE"].values[prof])

                    if juld is None or lat is None or lon is None:
                        print(f"Skipping profile {prof}: invalid date/location")
                        continue

                    description = (
                        f"Ocean profile at lat {lat}, lon {lon}, "
                        f"date {juld:%Y-%m-%d}"
                    )
                    embedding = model.encode(description).tolist()

                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO profiles
                                (n_prof, juld, latitude, longitude, embedding)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (prof, juld, lat, lon, embedding),
                        )
                        profile_id = cur.fetchone()[0]

                    level_rows = []
                    for level in range(ds.sizes["N_LEVELS"]):
                        pres = get_number(ds["PRES"].values[prof, level])
                        temp = get_number(ds["TEMP"].values[prof, level])
                        psal = get_number(ds["PSAL"].values[prof, level])

                        # Skip levels with no usable measurements.
                        if pres is None and temp is None and psal is None:
                            continue

                        level_rows.append(
                            (profile_id, level, pres, temp, psal, juld)
                        )

                    if level_rows:
                        with conn.cursor() as cur:
                            execute_batch(
                                cur,
                                """
                                INSERT INTO profile_levels
                                    (profile_id, n_levels, pres, temp, psal, juld)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                """,
                                level_rows,
                                page_size=500,
                            )

                    conn.commit()
                    print(
                        f"Imported profile {prof}: "
                        f"{len(level_rows)} measurement levels"
                    )

    print("Import complete!")


if __name__ == "__main__":
    main()
