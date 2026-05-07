import pandas as pd
import numpy as np

# NYC Airport Zone IDs
JFK_ID = 132
LGA_ID = 138
EWR_ID = 1

# JFK flat rate fare range
JFK_FLAT_RATE_MIN = 68
JFK_FLAT_RATE_MAX = 72

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def create_features(df, zone_coords):
    # 1. Parse datetime
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    # 2. Time features
    df['hour']         = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week']  = df['tpep_pickup_datetime'].dt.dayofweek
    df['day_of_month'] = df['tpep_pickup_datetime'].dt.day
    df['month']        = df['tpep_pickup_datetime'].dt.month

    # 3. Boolean time flags
    df['is_weekend']   = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].apply(
        lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
    )
    df['is_night'] = df['hour'].apply(
        lambda x: 1 if (x >= 22 or x <= 5) else 0
    )

    # 4. Merge pickup coordinates
    df = df.merge(
        zone_coords[['LocationID', 'latitude', 'longitude']].rename(columns={
            'LocationID': 'PULocationID',
            'latitude'  : 'pu_lat',
            'longitude' : 'pu_lon'
        }),
        on='PULocationID', how='left'
    )

    # 5. Merge dropoff coordinates
    df = df.merge(
        zone_coords[['LocationID', 'latitude', 'longitude']].rename(columns={
            'LocationID': 'DOLocationID',
            'latitude'  : 'do_lat',
            'longitude' : 'do_lon'
        }),
        on='DOLocationID', how='left'
    )

    # 6. Calculate haversine distance
    df['distance_km'] = haversine(
        df['pu_lat'], df['pu_lon'],
        df['do_lat'], df['do_lon']
    )

    # 7. Drop rows with missing distance
    df = df.dropna(subset=['distance_km'])
    df = df[df['distance_km'] > 0.1]

    # 8. ✅ NEW: Airport trip flags
    df['is_jfk_trip'] = (
        (df['PULocationID'] == JFK_ID) |
        (df['DOLocationID'] == JFK_ID)
    ).astype(int)

    df['is_lga_trip'] = (
        (df['PULocationID'] == LGA_ID) |
        (df['DOLocationID'] == LGA_ID)
    ).astype(int)

    df['is_ewr_trip'] = (
        (df['PULocationID'] == EWR_ID) |
        (df['DOLocationID'] == EWR_ID)
    ).astype(int)

    # Combined airport flag
    df['is_airport_trip'] = (
        (df['is_jfk_trip'] == 1) |
        (df['is_lga_trip'] == 1) |
        (df['is_ewr_trip'] == 1)
    ).astype(int)

    # 9. ✅ NEW: Remove JFK flat-rate fares — they confuse the model
    # These are fixed $70 fares regardless of distance
    jfk_flat_mask = (
        (df['is_jfk_trip'] == 1) &
        (df['fare_amount'] >= JFK_FLAT_RATE_MIN) &
        (df['fare_amount'] <= JFK_FLAT_RATE_MAX)
    )
    removed = jfk_flat_mask.sum()
    df = df[~jfk_flat_mask]
    print(f"🛫 Removed {removed} JFK flat-rate trips")

    # 10. ✅ NEW: Price per km — helps model understand fare efficiency
    df['fare_per_km'] = df['fare_amount'] / df['distance_km'].replace(0, np.nan)
    df['fare_per_km'] = df['fare_per_km'].clip(upper=20)  # remove insane outliers

    print(f"✅ Features created. Shape: {df.shape}")

    # Keep these columns from raw data
    surcharge_cols = ['extra', 'mta_tax', 'tolls_amount', 'improvement_surcharge']
    for col in surcharge_cols:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)  # remove negative surcharges

    return df