from xgboost import XGBRegressor
import joblib
from src.config import MODEL_PATH

def train_demand_model(X, y):
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH + "demand_model.pkl")
    print("Demand model saved")
    return model