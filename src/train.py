from xgboost import XGBRegressor
import joblib
from src.config import MODEL_PATH

def train_model(X, y, model_name):
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH + model_name)

    print(f"Model saved at {MODEL_PATH + model_name}")

    return model