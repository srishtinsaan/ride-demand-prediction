import pandas as pd

def create_features(df):
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])

    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['day_of_month'] = df['tpep_pickup_datetime'].dt.day

    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    df['is_rush_hour'] = df['hour'].apply(
        lambda x: 1 if (7 <= x <= 9) or (17 <= x <= 19) else 0
    )

    df['is_night'] = df['hour'].apply(
        lambda x: 1 if (x >= 22 or x <= 5) else 0
    )

    return df