import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from geopy.geocoders import Nominatim
from geopy.distance import distance
import re
import json

DB_CONFIG = {
    "host": "localhost",
    "database": "floatchatai",
    "user": "postgres",
    "password": "Owais@786"
}

TOP_K = 3          
RADIUS_METERS = 50_000 
model = SentenceTransformer('all-MiniLM-L6-v2')  

def get_embedding(text):
    """Convert user query to embedding vector."""
    return model.encode(text).tolist()

def cosine_similarity(a, b):
    """Compute cosine similarity between two lists."""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def geocode_place(place_name):
    """Convert a place name to (latitude, longitude) using Nominatim."""
    geolocator = Nominatim(user_agent="floatchat")
    location = geolocator.geocode(place_name)
    if location:
        return location.latitude, location.longitude
    return None, None

def extract_lat_lon(user_query):
    """Extract lat/long from query if present."""
    lat_match = re.search(r"lat\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    lon_match = re.search(r"long\s*=\s*([-+]?\d*\.?\d+)", user_query, re.IGNORECASE)
    if lat_match and lon_match:
        return float(lat_match.group(1)), float(lon_match.group(1))
    return None, None

def query_profiles(user_query):
    lat, lon = extract_lat_lon(user_query)

    if lat is None or lon is None:
        lat, lon = geocode_place(user_query)

    query_emb = get_embedding(user_query)

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT id, latitude, longitude, juld, embedding FROM profiles")
    profiles = cur.fetchall()
    print(f"Total profiles fetched: {len(profiles)}")

    sims = []
    for profile in profiles:
        profile_id, plat, plon, juld, emb = profile

        if isinstance(emb, str):
            emb = json.loads(emb)

        if lat is not None and lon is not None:
            dist = distance((lat, lon), (plat, plon)).meters
            if dist > RADIUS_METERS:
                continue
        else:
            dist = 0 

        sim = cosine_similarity(query_emb, emb)

       
        distance_score = 1 / (1 + dist)  
        combined_score = 0.7 * sim + 0.3 * distance_score

        sims.append((profile_id, plat, plon, juld))

    top_profiles = sorted(sims, key=lambda x: x[0], reverse=True)[:TOP_K]

    results = []
    for  profile_id, plat, plon, juld in top_profiles:
        cur.execute("""
            SELECT pres, temp, psal
            FROM profile_levels
            WHERE profile_id = %s
            ORDER BY n_levels ASC
        """, (profile_id,))
        levels = cur.fetchall()

        results.append({
            "profile_id": profile_id,
            "lat": plat,
            "lon": plon,
            "time": juld.strftime("%Y-%m-%d %H:%M:%S"),
            "depth_levels": [{"pres": l[0], "temp": l[1], "salinity": l[2]} for l in levels],
            "query_explain": f"Matched using weighted embedding similarity and proximity for: '{user_query}'"
        })

    conn.close()
    return results


if __name__ == "__main__":
    test_queries = [
        "Salt levels at lat=-43.037, long=130",
        "Salinity near mumbai",
    ]

    for query in test_queries:
        output = query_profiles(query)
        print(f"\nQuery: {query}")
        print(json.dumps(output, indent=2))
