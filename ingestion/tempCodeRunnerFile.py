
import os
import xarray as xr
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

DATA_DIR = "../data" 
BULK_SIZE = 1000     
DB_CONFIG = {
    "host": "localhost",
    "database": "floatchatai",
    "user": "postgres",
    "password": "Owais@786"
}

LAT_RANGE = (-10, 30)
TIME_START = "2023-01-01"
TIME_END = "2023-06-30"

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def preprocess_file(file_path):
    print(f"Processing {file_path}...")
    ds = xr.open_dataset(file_path, chunks={'time': 1000})
    df = ds.to_dataframe().reset_index()
    
    if 'lat' in df.columns:
        df = df[df['lat'].between(LAT_RANGE[0], LAT_RANGE[1])]
    if 'time' in df.columns:
        df = df[(df['time'] >= pd.Timestamp(TIME_START)) & (df['time'] <= pd.Timestamp(TIME_END))]
    
    columns = ['float_id', 'time', 'lat', 'lon', 'temp', 'salinity']
    df = df[[c for c in columns if c in df.columns]]
    return df

def insert_to_postgres(df, conn):
    cur = conn.cursor()
    sql = """
    INSERT INTO argo_data (float_id, time, lat, lon, temp, salinity)
    VALUES (%s,%s,%s,%s,%s,%s)
    """
    records = df.to_records(index=False)
    execute_batch(cur, sql, records, page_size=BULK_SIZE)
    conn.commit()
    cur.close()

def process_all():
    conn = connect_db()
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".nc")]
    
    for file in files:
        df = preprocess_file(os.path.join(DATA_DIR, file))
        if not df.empty:
            insert_to_postgres(df, conn)
            print(f"Inserted {len(df)} rows from {file}")
        else:
            print(f"No relevant data in {file}")
    
    conn.close()
    print("All files processed successfully!")

if __name__ == "__main__":
    process_all()
