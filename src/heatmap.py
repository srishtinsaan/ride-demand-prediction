import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import streamlit as st
import pandas as pd

def render_heatmap(demand_model, zones, hour, day):
    st.subheader("🔥 Demand Heatmap")

    results = []
    skipped = 0

    for zone in zones['LocationID'].unique():
        row = zones[zones['LocationID'] == zone].iloc[0]

        # 1. Skip zones with missing coordinates
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            skipped += 1
            continue

        # 2. Build input
        input_dict = {
            'PULocationID': zone,
            'hour': hour,
            'day_of_week': day
        }

        # 3. Predict demand with specific error handling
        try:
            pred = demand_model.predict(pd.DataFrame([input_dict]))[0]
            pred = max(0, float(pred))  # clamp negative predictions

            results.append([
                row['latitude'],
                row['longitude'],
                pred
            ])

        except Exception as e:
            skipped += 1
            print(f"⚠️  Skipped zone {zone}: {e}")
            continue

    # 4. Feedback
    print(f"✅ Heatmap built: {len(results)} zones plotted, {skipped} skipped")

    if len(results) == 0:
        st.warning("⚠️ No demand data available to show heatmap.")
        return

    # 5. Build map
    avg_lat = zones['latitude'].mean()
    avg_lon = zones['longitude'].mean()

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)
    HeatMap(results, radius=15, blur=20, min_opacity=0.4).add_to(m)

    # 6. Render
    st_folium(m, width=900, height=500)