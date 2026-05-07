import folium
from folium.plugins import HeatMap
import os

def generate_heatmap(data, save_path, hour=None, day=None):
    # 1. Guard against empty data
    if not data or len(data) == 0:
        print("⚠️  No data provided to generate heatmap")
        return None

    # 2. Auto-create output directory if missing
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 3. Build map centered on NYC
    m = folium.Map(
        location=[40.7128, -74.0060],
        zoom_start=11,
        tiles="CartoDB positron"  # cleaner look than default
    )

    # 4. Add heatmap layer with proper settings
    HeatMap(
        data,
        radius=15,
        blur=20,
        min_opacity=0.4,
        max_zoom=13
    ).add_to(m)

    # 5. Add title if hour/day provided
    if hour is not None and day is not None:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        title = f"Demand Heatmap — {days[day]} at {hour}:00"
        title_html = f"""
            <h3 style="position:fixed; top:10px; left:50%; transform:translateX(-50%);
                       z-index:1000; background:white; padding:6px 12px;
                       border-radius:8px; box-shadow:2px 2px 6px rgba(0,0,0,0.3);">
                {title}
            </h3>
        """
        m.get_root().html.add_child(folium.Element(title_html))

    # 6. Save
    m.save(save_path)
    print(f"💾 Heatmap saved at: {save_path}")

    return m