from src.data_loader import load_data
from src.feature_engineering import create_features
from src.train import train_model
from src.predict import load_model, predict
from src.demand_model import train_demand_model
from src.surge import calculate_surge
from src.visualization import generate_heatmap

import pandas as pd
import os


# ================= TRAINING =================
def run_training():
    print("\n🚀 Starting Training Pipeline...\n")
    df, zones = load_data()

    # ✅ Pass zone_coords to create_features (needed for distance_km)
    zone_coords = pd.read_csv("data/zone_coords.csv")
    df = create_features(df, zone_coords)

    # ===== FARE MODEL =====
    fare_features = [
    'PULocationID', 'DOLocationID',
    'hour', 'day_of_week', 'day_of_month', 'month',
    'trip_distance', 'distance_km',
    'passenger_count',
    'is_weekend', 'is_rush_hour', 'is_night',
    'is_jfk_trip', 'is_lga_trip', 'is_ewr_trip', 'is_airport_trip',
    'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge'  
]

    fare_df = df[fare_features + ['total_amount']].dropna()
    X_fare = fare_df.drop('total_amount', axis=1)
    y_fare = fare_df['total_amount']

    print("🎯 Training Fare Model...")
    train_model(X_fare, y_fare, "fare_model.pkl")

    # ===== DEMAND MODEL =====
    print("\n🎯 Training Demand Model...")
    demand_df = df.groupby(
        ['PULocationID', 'hour', 'day_of_week']
    ).size().reset_index(name='trip_count')

    X_demand = demand_df[['PULocationID', 'hour', 'day_of_week']]
    y_demand = demand_df['trip_count']

    train_demand_model(X_demand, y_demand)

    print("\n✅ Training complete!\n")

    # ✅ Return demand_df so run_prediction() can reuse it
    return demand_df


# ================= PREDICTION =================
def run_prediction(demand_df=None):
    print("\n🔍 Running Prediction...\n")

    fare_model   = load_model("fare_model.pkl")
    demand_model = load_model("demand_model.pkl")

    input_data = {
    'PULocationID'          : 161,
    'DOLocationID'          : 132,
    'hour'                  : 18,
    'day_of_week'           : 4,
    'day_of_month'          : 1,
    'month'                 : 1,
    'trip_distance'         : 12,
    'distance_km'           : 15.3,
    'passenger_count'       : 1,
    'is_weekend'            : 0,
    'is_rush_hour'          : 1,
    'is_night'              : 0,
    'is_jfk_trip'           : 1,   # DOLocationID 132 = JFK
    'is_lga_trip'           : 0,
    'is_ewr_trip'           : 0,
    'is_airport_trip'       : 1,
    'extra'                 : 0.5,
    'mta_tax'               : 0.5,
    'tolls_amount'          : 0.0,
    'improvement_surcharge' : 0.3
}
    # Fare prediction
    base_fare = predict(fare_model, input_data)

    # Demand prediction
    demand_input = {
        'PULocationID': input_data['PULocationID'],
        'hour'        : input_data['hour'],
        'day_of_week' : input_data['day_of_week']
    }
    predicted_demand = predict(demand_model, demand_input)

    # ✅ Reuse demand_df instead of reloading entire dataset
    if demand_df is None:
        print("⚠️  demand_df not passed — loading fresh (slow path)")
        df, _ = load_data()
        zone_coords = pd.read_csv("data/zone_coords.csv")
        df = create_features(df, zone_coords)
        demand_df = df.groupby(
            ['PULocationID', 'hour', 'day_of_week']
        ).size().reset_index(name='trip_count')

    avg_demand = demand_df[
        (demand_df['PULocationID'] == input_data['PULocationID']) &
        (demand_df['hour']         == input_data['hour']) &
        (demand_df['day_of_week']  == input_data['day_of_week'])
    ]['trip_count'].mean()

    if pd.isna(avg_demand):
        avg_demand = demand_df['trip_count'].mean()
        print("⚠️  Zone not found in demand data — using global average")

    # Surge calculation
    surge_multiplier, surge_label, ratio = calculate_surge(
        predicted_demand, avg_demand
    )
    final_price = base_fare * surge_multiplier

    print("=" * 45)
    print(f"  Base Fare        : ${base_fare:.2f}")
    print(f"  Predicted Demand : {predicted_demand:.2f}")
    print(f"  Avg Demand       : {avg_demand:.2f}")
    print(f"  Demand Ratio     : {ratio:.2f}")
    print(f"  Surge Label      : {surge_label}")
    print(f"  Surge Multiplier : {surge_multiplier}x")
    print(f"  Final Price      : ${final_price:.2f}")
    print("=" * 45)


# ================= HEATMAP =================
def generate_demand_map(zones, demand_model, hour=18, day=4):
    print(f"\n🗺️  Generating heatmap for hour={hour}, day={day}...\n")

    results = []
    skipped = 0

    for zone in zones['LocationID'].unique():
        try:
            pred = predict(demand_model, {
                'PULocationID': zone,
                'hour'        : hour,
                'day_of_week' : day
            })
            pred = max(0, float(pred))
            results.append({
                'LocationID'       : zone,
                'predicted_demand' : pred
            })
        except Exception as e:
            skipped += 1
            print(f"⚠️  Skipped zone {zone}: {e}")
            continue

    print(f"✅ {len(results)} zones predicted, {skipped} skipped")

    pred_df = pd.DataFrame(results)

    # Merge coordinates
    pred_df = pred_df.merge(
        zones[['LocationID', 'latitude', 'longitude']],
        on='LocationID', how='left'
    )
    pred_df = pred_df.dropna(subset=['latitude', 'longitude'])

    # Surge flag
    avg_demand = pred_df['predicted_demand'].mean()
    pred_df['surge'] = pred_df['predicted_demand'] > (1.5 * avg_demand)

    # Generate and save heatmap
    save_path = "outputs/maps/heatmap.html"
    generate_heatmap(
        data=pred_df[['latitude', 'longitude', 'predicted_demand']].values.tolist(),
        save_path=save_path,
        hour=hour,
        day=day
    )

    return pred_df


# ================= MAIN =================
if __name__ == "__main__":
    demand_df = run_training()
    run_prediction(demand_df)