import os
IS_CLOUD = os.environ.get("IS_STREAMLIT_CLOUD", "false") == "true"

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from src.predict import load_model, predict
from src.surge import calculate_surge

# ================= CONFIG =================
st.set_page_config(page_title="🚖 Ride Pricing", layout="wide")

# ================= SESSION =================
if "result" not in st.session_state:
    st.session_state.result = None

# ================= FEATURES =================
FEATURES = [
    'PULocationID', 'DOLocationID',
    'hour', 'day_of_week', 'day_of_month', 'month',
    'trip_distance', 'distance_km',
    'passenger_count',
    'is_weekend', 'is_rush_hour', 'is_night',
    'is_jfk_trip', 'is_lga_trip', 'is_ewr_trip', 'is_airport_trip',
    'extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge'
]

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    fare_model   = load_model("fare_model.pkl")
    demand_model = load_model("demand_model.pkl")
    return fare_model, demand_model

# ================= LOAD DATA =================
@st.cache_data
def load_app_data():
    zones  = pd.read_csv("data/raw/taxi_zone_lookup.csv")
    coords = pd.read_csv("data/zone_coords.csv")
    zones  = zones.merge(coords, on="LocationID", how="left")
    zones['full_name'] = zones['Borough'] + " – " + zones['Zone']
    return zones

@st.cache_data
def load_demand_baseline():
    # ✅ Loads only pre-saved CSV — not full parquet
    return pd.read_csv("data/demand_baseline.csv")

@st.cache_data
def load_eval_sample():
    # ✅ Loads only 10K row sample for evaluation
    return pd.read_csv("data/eval_sample.csv")

# ✅ Clean single load — no double call, no tuple unpacking
fare_model, demand_model = load_models()
zones                    = load_app_data()
temp_df                  = load_demand_baseline()

# ================= HAVERSINE =================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a    = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def nearby(lat, lon, center_lat, center_lon, radius_km=6):
    dlat = np.radians(lat - center_lat)
    dlon = np.radians(lon - center_lon)
    a    = np.sin(dlat/2)**2 + np.cos(np.radians(center_lat)) * np.cos(np.radians(lat)) * np.sin(dlon/2)**2
    return 6371 * 2 * np.arcsin(np.sqrt(a)) <= radius_km

# ================= UI =================
st.title("🚖 Smart Ride Pricing System")

DAY_OPTIONS = {
    "Monday"    : 0,
    "Tuesday"   : 1,
    "Wednesday" : 2,
    "Thursday"  : 3,
    "Friday"    : 4,
    "Saturday"  : 5,
    "Sunday"    : 6
}

with st.form("ride_form"):
    col1, col2 = st.columns(2)

    with col1:
        pu_zone    = st.selectbox("📍 Pickup Location", zones['full_name'])
        t1, t2, t3 = st.columns(3)
        with t1:
            hour_12 = st.selectbox("🕐 Hour", list(range(1, 13)), index=5)
        with t2:
            minute  = st.selectbox("⏱ Minute", [0, 15, 30, 45], index=0)
        with t3:
            am_pm   = st.selectbox("🌅 AM/PM", ["AM", "PM"], index=1)
        passengers = st.number_input("👤 Passengers", min_value=1, max_value=6, value=1)

    with col2:
        do_zone  = st.selectbox("🏁 Drop Location", zones['full_name'])
        day_name = st.selectbox("📅 Day of Week", list(DAY_OPTIONS.keys()), index=3)

    submit = st.form_submit_button("🚀 Predict Price")

# ================= PREDICTION =================
if submit:
    if am_pm == "AM":
        hour = hour_12 % 12
    else:
        hour = (hour_12 % 12) + 12

    day    = DAY_OPTIONS[day_name]
    pu_row = zones[zones['full_name'] == pu_zone].iloc[0]
    do_row = zones[zones['full_name'] == do_zone].iloc[0]
    pu_id  = int(pu_row['LocationID'])
    do_id  = int(do_row['LocationID'])

    if pu_id == do_id:
        st.warning("⚠️ Pickup and Drop must be different zones.")
        st.stop()

    if pd.isna(pu_row['latitude']) or pd.isna(do_row['latitude']):
        st.error("❌ Coordinates missing for selected zone. Try another.")
        st.stop()

    distance_km     = round(haversine(
        pu_row['latitude'], pu_row['longitude'],
        do_row['latitude'], do_row['longitude']
    ), 3)
    is_weekend      = 1 if day >= 5 else 0
    is_rush_hour    = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    is_night        = 1 if (hour >= 22 or hour <= 5) else 0
    day_of_month    = pd.Timestamp.now().day
    month           = pd.Timestamp.now().month
    is_jfk_trip     = 1 if pu_id == 132 or do_id == 132 else 0
    is_lga_trip     = 1 if pu_id == 138 or do_id == 138 else 0
    is_ewr_trip     = 1 if pu_id == 1   or do_id == 1   else 0
    is_airport_trip = 1 if (is_jfk_trip or is_lga_trip or is_ewr_trip) else 0

    input_data = {
        'PULocationID'          : pu_id,
        'DOLocationID'          : do_id,
        'hour'                  : hour,
        'day_of_week'           : day,
        'day_of_month'          : day_of_month,
        'month'                 : month,
        'trip_distance'         : distance_km,
        'distance_km'           : distance_km,
        'passenger_count'       : passengers,
        'is_weekend'            : is_weekend,
        'is_rush_hour'          : is_rush_hour,
        'is_night'              : is_night,
        'is_jfk_trip'           : is_jfk_trip,
        'is_lga_trip'           : is_lga_trip,
        'is_ewr_trip'           : is_ewr_trip,
        'is_airport_trip'       : is_airport_trip,
        'extra'                 : 0.5,
        'mta_tax'               : 0.5,
        'tolls_amount'          : 0.0,
        'improvement_surcharge' : 0.3
    }

    base_fare        = predict(fare_model, input_data)
    predicted_demand = predict(demand_model, {
        'PULocationID': pu_id,
        'hour'        : hour,
        'day_of_week' : day
    })

    avg_demand = temp_df[
        (temp_df['PULocationID'] == pu_id) &
        (temp_df['hour']         == hour)  &
        (temp_df['day_of_week']  == day)
    ]['trip_count'].mean()

    if pd.isna(avg_demand):
        avg_demand = temp_df['trip_count'].mean()

    surge, surge_label, ratio = calculate_surge(predicted_demand, avg_demand)
    final_price = round(base_fare * surge, 2)

    st.session_state.result = {
        "base_fare"        : round(base_fare, 2),
        "surge"            : surge,
        "surge_label"      : surge_label,
        "ratio"            : ratio,
        "final_price"      : final_price,
        "predicted_demand" : round(predicted_demand, 2),
        "avg_demand"       : round(avg_demand, 2),
        "distance"         : distance_km,
        "pu_lat"           : pu_row['latitude'],
        "pu_lon"           : pu_row['longitude'],
        "pu_name"          : pu_zone,
        "do_lat"           : do_row['latitude'],
        "do_lon"           : do_row['longitude'],
        "do_name"          : do_zone,
        "hour"             : hour,
        "day"              : day
    }

# ================= DISPLAY =================
if st.session_state.result:
    r = st.session_state.result

    st.subheader("💰 Pricing Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Base Fare",  f"${r['base_fare']:.2f}")
    c2.metric("Demand",      r['surge_label'])
    c3.metric("Surge",       f"{r['surge']}x")
    c4.metric("Final Price", f"${r['final_price']:.2f}")
    st.write(f"📏 Distance: **{r['distance']:.2f} km** | Demand Ratio: **{r['ratio']:.2f}**")

    # ===== Route Map =====
    st.subheader("🗺️ Route Map")
    mid_lat   = (r['pu_lat'] + r['do_lat']) / 2
    mid_lon   = (r['pu_lon'] + r['do_lon']) / 2
    route_map = folium.Map(location=[mid_lat, mid_lon], zoom_start=12)
    folium.Marker(
        [r['pu_lat'], r['pu_lon']],
        tooltip=f"📍 Pickup: {r['pu_name']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(route_map)
    folium.Marker(
        [r['do_lat'], r['do_lon']],
        tooltip=f"🏁 Drop: {r['do_name']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(route_map)
    st_folium(route_map, width=900, height=400)

    # ===== Demand Heatmap =====
    st.subheader("🔥 Demand Heatmap")

    mid_lat = (r['pu_lat'] + r['do_lat']) / 2
    mid_lon = (r['pu_lon'] + r['do_lon']) / 2

    all_circle_data = []
    for _, zone_row in zones.iterrows():
        if pd.isna(zone_row['latitude']):
            continue
        try:
            pred = predict(demand_model, {
                'PULocationID': zone_row['LocationID'],
                'hour'        : r['hour'],
                'day_of_week' : r['day']
            })
            all_circle_data.append({
                'lat'        : zone_row['latitude'],
                'lon'        : zone_row['longitude'],
                'demand'     : max(0, float(pred)),
                'location_id': int(zone_row['LocationID'])
            })
        except:
            continue

    circle_data  = [d for d in all_circle_data if nearby(d['lat'], d['lon'], mid_lat, mid_lon)]
    heatmap_data = [[d['lat'], d['lon'], d['demand']] for d in circle_data]

    h_map = folium.Map(
        location=[mid_lat, mid_lon],
        zoom_start=13,
        tiles="CartoDB positron"
    )

    if heatmap_data:
        HeatMap(heatmap_data, radius=15, blur=20, min_opacity=0.3).add_to(h_map)

    folium.Marker(
        [r['pu_lat'], r['pu_lon']],
        tooltip=f"📍 Pickup: {r['pu_name']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(h_map)
    folium.Marker(
        [r['do_lat'], r['do_lon']],
        tooltip=f"🏁 Drop: {r['do_name']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(h_map)

    if circle_data:
        global_avg = sum(d['demand'] for d in circle_data) / len(circle_data)

        for d in circle_data:
            is_surge  = d['demand'] > (1.5 * global_avg)
            color     = 'red' if is_surge else 'green'
            zone_info = zones[zones['LocationID'] == d['location_id']]
            zone_name = zone_info['Zone'].iloc[0]    if len(zone_info) > 0 else "Unknown"
            borough   = zone_info['Borough'].iloc[0] if len(zone_info) > 0 else "Unknown"

            folium.CircleMarker(
                location=[d['lat'], d['lon']],
                radius=6,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.35,
                dash_array='4 4',
                opacity=0.85,
                popup=folium.Popup(
                    f"<b>{zone_name}</b><br>"
                    f"Borough: {borough}<br>"
                    f"Predicted trips: {int(d['demand'])}<br>"
                    f"Surge: {'YES 🔴' if is_surge else 'NO 🟢'}",
                    max_width=200
                )
            ).add_to(h_map)

    st_folium(h_map, width=900, height=500)

# ================= MODEL EVALUATION =================
st.subheader("🧪 Model Evaluation")

if IS_CLOUD:
    # ✅ On cloud — use pre-saved 10K sample only
    if st.button("Run Evaluation"):
        try:
            eval_df = load_eval_sample()
        except FileNotFoundError:
            st.error("❌ eval_sample.csv not found. Run generate_eval_sample.py locally first.")
            st.stop()

        eval_df = eval_df.dropna(subset=FEATURES + ['total_amount'])
        eval_df = eval_df[
            (eval_df['total_amount'] >= 3.0) &
            (eval_df['total_amount'] <= 200)
        ]

        X      = eval_df[FEATURES]
        y      = eval_df['total_amount']
        y_pred = fare_model.predict(X)

        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2   = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"${mae:.2f}")
        m2.metric("RMSE", f"${rmse:.2f}")
        m3.metric("R²",   f"{r2:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y, y_pred, alpha=0.3, s=10, color='steelblue')
        ax.plot([3, 200], [3, 200], 'r--', linewidth=1.5, label='Perfect Prediction')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_xlabel("Actual Total ($)")
        ax.set_ylabel("Predicted Total ($)")
        ax.set_title("Model Accuracy — Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🔍 Sample Predictions")
        sample  = eval_df.sample(min(10, len(eval_df)))
        samples = []
        for _, row in sample.iterrows():
            pred = predict(fare_model, row[FEATURES].to_dict())
            samples.append({
                "Actual Total" : f"${row['total_amount']:.2f}",
                "Predicted"    : f"${pred:.2f}",
                "Difference"   : f"${abs(row['total_amount'] - pred):.2f}"
            })
        st.dataframe(pd.DataFrame(samples))

else:
    # ✅ Local — same evaluation block, no restrictions
    if st.button("Run Evaluation"):
        try:
            eval_df = load_eval_sample()
        except FileNotFoundError:
            st.error("❌ eval_sample.csv not found. Run generate_eval_sample.py first.")
            st.stop()

        eval_df = eval_df.dropna(subset=FEATURES + ['total_amount'])
        eval_df = eval_df[
            (eval_df['total_amount'] >= 3.0) &
            (eval_df['total_amount'] <= 200)
        ]

        X      = eval_df[FEATURES]
        y      = eval_df['total_amount']
        y_pred = fare_model.predict(X)

        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2   = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE",  f"${mae:.2f}")
        m2.metric("RMSE", f"${rmse:.2f}")
        m3.metric("R²",   f"{r2:.4f}")

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y, y_pred, alpha=0.3, s=10, color='steelblue')
        ax.plot([3, 200], [3, 200], 'r--', linewidth=1.5, label='Perfect Prediction')
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_xlabel("Actual Total ($)")
        ax.set_ylabel("Predicted Total ($)")
        ax.set_title("Model Accuracy — Actual vs Predicted")
        ax.legend()
        st.pyplot(fig)

        st.subheader("🔍 Sample Predictions")
        sample  = eval_df.sample(min(10, len(eval_df)))
        samples = []
        for _, row in sample.iterrows():
            pred = predict(fare_model, row[FEATURES].to_dict())
            samples.append({
                "Actual Total" : f"${row['total_amount']:.2f}",
                "Predicted"    : f"${pred:.2f}",
                "Difference"   : f"${abs(row['total_amount'] - pred):.2f}"
            })
        st.dataframe(pd.DataFrame(samples))