import folium
from folium.plugins import HeatMap
import webbrowser
import os


def generate_heatmap(pred_df, save_path="outputs/maps/heatmap.html"):
    # ===== Create base map =====
    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="CartoDB positron"
    )

    # ===== Prepare heatmap data =====
    heat_data = [
        [
            float(row['latitude']),
            float(row['longitude']),
            float(row['predicted_demand'])
        ]
        for _, row in pred_df.iterrows()
    ]

    # ===== Improved Heatmap (sharper + realistic) =====
    HeatMap(
        heat_data,
        radius=20,     # smaller radius → sharper
        blur=15,       # less blur → clearer hotspots
        min_opacity=0.3
    ).add_to(m)

    # ===== Add dynamic markers =====
    for _, row in pred_df.iterrows():
        demand = float(row['predicted_demand'])

        # 🔥 Dynamic radius based on demand
        radius = max(5, min(12, demand / 50))

        # 🎯 Surge color logic
        color = "red" if row['surge'] else "green"

        folium.CircleMarker(
            location=[float(row['latitude']), float(row['longitude'])],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{row['Zone']}</b><br>"
                f"Demand: {int(demand)}<br>"
                f"Surge: {'YES 🔴' if row['surge'] else 'NO 🟢'}",
                max_width=200
            )
        ).add_to(m)

    # ===== Ensure output folder =====
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ===== Save map =====
    m.save(save_path)
    print(f"✅ Heatmap saved at: {save_path}")

    # ===== Auto open in browser =====
    webbrowser.open('file://' + os.path.realpath(save_path))

    return m