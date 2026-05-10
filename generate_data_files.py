# generate_data_files.py  — run this ONCE locally
from src.data_loader import load_data
from src.feature_engineering import create_features
import pandas as pd

FEATURES = [
    'PULocationID', 'DOLocationID',
    'hour', 'day_of_week', 'day_of_month', 'month',
    'trip_distance', 'distance_km',
    'passenger_count',
    'is_weekend', 'is_rush_hour', 'is_night',
    'is_jfk_trip', 'is_lga_trip', 'is_ewr_trip', 'is_airport_trip',
    'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge'
]

print("📂 Loading data...")
zone_coords = pd.read_csv("data/zone_coords.csv")
df, _       = load_data()

print("⚙️  Creating features...")
df = create_features(df, zone_coords)

# ✅ FILE 1 — demand_baseline.csv
print("📊 Generating demand_baseline.csv...")
demand_df = df.groupby(
    ['PULocationID', 'hour', 'day_of_week']
).size().reset_index(name='trip_count')
demand_df.to_csv("data/demand_baseline.csv", index=False)
print(f"✅ demand_baseline.csv saved — {len(demand_df)} rows")

# ✅ FILE 2 — eval_sample.csv
print("📊 Generating eval_sample.csv...")
eval_df = df.dropna(subset=FEATURES + ['total_amount'])
eval_df = eval_df[
    (eval_df['total_amount'] >= 3.0) &
    (eval_df['total_amount'] <= 200)
]
eval_df.sample(10000, random_state=42)[FEATURES + ['total_amount']].to_csv(
    "data/eval_sample.csv", index=False
)

