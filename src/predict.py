import joblib
import pandas as pd
import numpy as np
from src.config import MODEL_PATH

def load_model(model_name):
    return joblib.load(MODEL_PATH + model_name)

def predict(model, input_dict):
    # 1. Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # 2. Check for missing values
    if df.isnull().any().any():
        missing = df.columns[df.isnull().any()].tolist()
        raise ValueError(f"❌ Missing values in input: {missing}")

    # 3. Predict
    prediction = model.predict(df)[0]

    # 4. Sanity check on output
    if prediction < 0:
        print(f"⚠️  WARNING: Negative prediction ({prediction:.2f}) — clamping to minimum fare $2.50")
        prediction = 2.50

    if prediction > 500:
        print(f"⚠️  WARNING: Unusually high prediction (${prediction:.2f}) — might be an outlier input")

    return round(float(prediction), 2)


def predict_with_surge(fare_model, demand_model, input_dict):
    # Step 1: Predict base fare
    base_fare = predict(fare_model, input_dict)

    # Step 2: Predict demand
    demand_score = predict(demand_model, input_dict)

    # Step 3: Apply surge multiplier
    if demand_score >= 0.8:
        surge = 1.8
        demand_label = "🔴 Very High"
    elif demand_score >= 0.6:
        surge = 1.4
        demand_label = "🟠 High"
    elif demand_score >= 0.4:
        surge = 1.1
        demand_label = "🟡 Moderate"
    else:
        surge = 1.0
        demand_label = "🟢 Normal"

    final_fare = round(base_fare * surge, 2)

    return {
        "base_fare": base_fare,
        "demand_label": demand_label,
        "demand_score": round(float(demand_score), 3),
        "surge_multiplier": surge,
        "final_fare": final_fare
    }