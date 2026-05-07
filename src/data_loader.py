import pandas as pd
from src.config import DATA_PATH

def load_data():
    # Load raw data
    df = pd.read_parquet(DATA_PATH + "yellow_tripdata_2026-01.parquet")
    zones = pd.read_csv(DATA_PATH + "taxi_zone_lookup.csv")

    # 1. Drop nulls
    df = df.dropna()

    # 2. based on 
    df = df[(df['total_amount'] >= 3.0) & (df['total_amount'] <= 200)]

    # 3. Remove invalid passenger count
    df = df[df['passenger_count'] >= 1]

    # 4. Remove zero/very short distance trips
    df = df[df['trip_distance'] > 0.1]

    # 5. Remove same zone trips
    df = df[df['PULocationID'] != df['DOLocationID']]

    # 6. Parse datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # Debug: show how much data survived
    print(f"✅ Cleaned data shape: {df.shape}")

    return df, zones