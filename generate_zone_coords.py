import geopandas as gpd
import pandas as pd
import os

def generate_zone_coords(
    shapefile_path="taxi_zones/taxi_zones.shp",
    output_path="data/zone_coords.csv"
):
    # 1. Check shapefile exists
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"❌ Shapefile not found at: {shapefile_path}")

    # 2. Load shapefile
    print("📂 Loading shapefile...")
    gdf = gpd.read_file(shapefile_path)
    print(f"   Loaded {len(gdf)} zones")

    # 3. Fix CRS and calculate centroids
    gdf = gdf.to_crs(epsg=3857)
    centroids = gdf.geometry.centroid
    centroids = gpd.GeoSeries(centroids, crs="EPSG:3857").to_crs(epsg=4326)

    gdf['latitude']  = centroids.y
    gdf['longitude'] = centroids.x

    # 4. Keep only needed columns
    df = gdf[['LocationID', 'latitude', 'longitude']].copy()

    # 5. Sanity check — no nulls
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"⚠️  Warning: {null_count} null values found in coordinates")
        df = df.dropna()

    # 6. Sanity check — valid NYC coordinate range
    df = df[
        (df['latitude'].between(40.4, 41.0)) &
        (df['longitude'].between(-74.4, -73.6))
    ]
    print(f"✅ {len(df)} valid zones after coordinate filtering")

    # 7. Auto-create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 8. Save
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")

    return df


if __name__ == "__main__":
    generate_zone_coords()