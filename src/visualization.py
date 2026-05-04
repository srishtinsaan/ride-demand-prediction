import folium
from folium.plugins import HeatMap

def generate_heatmap(data, save_path):
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

    HeatMap(data).add_to(m)
    m.save(save_path)

    return m