import os
import re
import json
import math
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from geopy.distance import distance

# Load root .env regardless of where this file is invoked from.
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DB_CONFIG = {
    "host": os.getenv("HOST_NAME", "localhost"),
    "database": os.getenv("DATABASE_NAME", "floatchatai"),
    "user": os.getenv("USER_NAME"),
    "password": os.getenv("PASSWORD"),
    "port": int(os.getenv("PORT", "5432")),
}

TOP_K = 3
RADIUS_METERS = 50_000
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text):
    """Convert user query to embedding vector."""
    return model.encode(text).tolist()


def cosine_similarity(a, b):
    """Compute cosine similarity between two lists."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def geocode_place(place_name):
    """Convert a place name to (latitude, longitude) using Nominatim."""
    try:
        geolocator = Nominatim(user_agent="floatchat")
        location = geolocator.geocode(place_name)
        if location:
            return location.latitude, location.longitude
    except Exception:
        pass
    return None, None


def extract_lat_lon(user_query):
    """Extract lat/long from query if present."""
    lat_match = re.search(r"lat\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    lon_match = re.search(r"long\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    if lat_match and lon_match:
        return float(lat_match.group(1)), float(lon_match.group(1))
    return None, None


def query_profiles(user_query):
    """Query profiles using embedding similarity and optional geographic proximity."""
    lat, lon = extract_lat_lon(user_query)
    if lat is None or lon is None:
        lat, lon = geocode_place(user_query)

    query_emb = get_embedding(user_query)

    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT id, latitude, longitude, juld, embedding FROM profiles")
        profiles = cur.fetchall()

        sims = []
        for profile in profiles:
            profile_id, plat, plon, juld, emb = profile
            if isinstance(emb, str):
                emb = json.loads(emb)
            if emb is None:
                continue

            if lat is not None and lon is not None:
                dist = distance((lat, lon), (plat, plon)).meters
                if dist > RADIUS_METERS:
                    continue
            else:
                dist = 0

            sim = cosine_similarity(query_emb, emb)
            distance_score = 1 / (1 + dist)
            combined_score = 0.7 * sim + 0.3 * distance_score
            sims.append((combined_score, profile_id, plat, plon, juld))

        top_profiles = sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_K]

        results = []
        for _, profile_id, plat, plon, juld in top_profiles:
            cur.execute(
                """
                SELECT pres, temp, psal
                FROM profile_levels
                WHERE profile_id = %s
                ORDER BY n_levels ASC
                """,
                (profile_id,),
            )
            levels = cur.fetchall()

            depth_levels = []
            for pres, temp, sal in levels:
                if pres is None or temp is None or sal is None:
                    continue
                if math.isnan(pres) or math.isnan(temp) or math.isnan(sal):
                    continue
                depth_levels.append({"pres": pres, "temp": temp, "salinity": sal})

            results.append(
                {
                    "profile_id": profile_id,
                    "lat": plat,
                    "lon": plon,
                    "time": juld.strftime("%Y-%m-%d %H:%M:%S"),
                    "depth_levels": depth_levels,
                    "query_explain": (
                        f"Matched using weighted embedding similarity and "
                        f"proximity for: '{user_query}'"
                    ),
                }
            )

        return results
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    test_queries = [
        "Salt levels at lat=-43.037, long=130",
        "Salinity near mumbai",
    ]

    for q in test_queries:
        output = query_profiles(q)
        print(f"\nQuery: {q}")
        print(json.dumps(output, indent=2))
