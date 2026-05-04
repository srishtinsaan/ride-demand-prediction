from src.data_loader import load_data
from src.feature_engineering import create_features
from src.train import train_model
from src.predict import load_model, predict
from src.demand_model import train_demand_model
from src.surge import calculate_surge
from src.heatmap import generate_heatmap

import geopandas as gpd
import pandas as pd


# ================= TRAINING =================
def run_training():
    df, zones = load_data()
    df = create_features(df)

    # ===== FARE MODEL =====
    fare_df = df[[
        'PULocationID',
        'DOLocationID',
        'hour',
        'day_of_week',
        'trip_distance',
        'passenger_count',
        'fare_amount'
    ]].dropna()

    X_fare = fare_df.drop('fare_amount', axis=1)
    y_fare = fare_df['fare_amount']

    train_model(X_fare, y_fare, "fare_model.pkl")

    # ===== DEMAND MODEL =====
    demand_df = df.groupby(
        ['PULocationID', 'hour', 'day_of_week']
    ).size().reset_index(name='trip_count')

    X_demand = demand_df[['PULocationID', 'hour', 'day_of_week']]
    y_demand = demand_df['trip_count']

    train_demand_model(X_demand, y_demand)


# ================= PREDICTION =================
def run_prediction():
    fare_model = load_model("fare_model.pkl")
    demand_model = load_model("demand_model.pkl")

    input_data = {
        'PULocationID': 161,
        'DOLocationID': 132,
        'hour': 18,
        'day_of_week': 4,
        'trip_distance': 12,
        'passenger_count': 1
    }

    # ===== Fare Prediction =====
    base_fare = predict(fare_model, input_data)

    # ===== Demand Prediction =====
    demand_input = {
        'PULocationID': input_data['PULocationID'],
        'hour': input_data['hour'],
        'day_of_week': input_data['day_of_week']
    }

    predicted_demand = predict(demand_model, demand_input)

    # ===== Load + Features =====
    df, zones = load_data()
    df = create_features(df)

    temp_df = df.groupby(
        ['PULocationID', 'hour', 'day_of_week']
    ).size().reset_index(name='trip_count')

    avg_demand = temp_df[
        (temp_df['PULocationID'] == input_data['PULocationID']) &
        (temp_df['hour'] == input_data['hour']) &
        (temp_df['day_of_week'] == input_data['day_of_week'])
    ]['trip_count'].mean()

    if pd.isna(avg_demand):
        avg_demand = temp_df['trip_count'].mean()

    # ===== Surge =====
    surge_multiplier = calculate_surge(predicted_demand, avg_demand)
    final_price = base_fare * surge_multiplier

    print("\n" + "="*45)
    print(f"Base Fare:        ${base_fare:.2f}")
    print(f"Predicted Demand: {predicted_demand:.2f}")
    print(f"Avg Demand:       {avg_demand:.2f}")
    print(f"Surge Multiplier: {surge_multiplier}x")
    print(f"Final Price:      ${final_price:.2f}")
    print("="*45)

    # ===== Heatmap =====
    generate_demand_map(zones, demand_model)


# ================= HEATMAP =================
def generate_demand_map(zones, demand_model):
    results = []

    for zone in zones['LocationID'].unique():
        try:
            pred = predict(demand_model, {
                'PULocationID': zone,
                'hour': 18,
                'day_of_week': 4
            })

            results.append({
                'LocationID': zone,
                'predicted_demand': float(pred)
            })
        except:
            continue

    pred_df = pd.DataFrame(results)

    # ===== Load shapefile =====
    gdf = gpd.read_file("taxi_zones/taxi_zones.shp")

    # ===== CORRECT CRS FIX (IMPORTANT) =====
    gdf = gdf.to_crs(epsg=4326)

    gdf['latitude'] = gdf.geometry.centroid.y
    gdf['longitude'] = gdf.geometry.centroid.x

    gdf = gdf[['LocationID', 'latitude', 'longitude']]

    # ===== Merge =====
    pred_df = pred_df.merge(gdf, on='LocationID', how='left')
    pred_df = pred_df.merge(
        zones[['LocationID', 'Zone']],
        on='LocationID',
        how='left'
    )

    # ===== Remove invalid coords =====
    pred_df = pred_df.dropna(subset=['latitude', 'longitude'])

    # ===== Surge zones =====
    avg_demand = pred_df['predicted_demand'].mean()
    pred_df['surge'] = pred_df['predicted_demand'] > (1.5 * avg_demand)

    # ===== Generate map =====
    generate_heatmap(pred_df)

    return pred_df


# ================= MAIN =================
if __name__ == "__main__":
    run_training()
    run_prediction()