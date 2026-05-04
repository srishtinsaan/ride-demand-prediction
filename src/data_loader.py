import pandas as pd
from src.config import DATA_PATH

def load_data():
    df = pd.read_parquet(DATA_PATH + "yellow_tripdata_2026-01.parquet")
    zones = pd.read_csv(DATA_PATH + "taxi_zone_lookup.csv")
    return df, zones