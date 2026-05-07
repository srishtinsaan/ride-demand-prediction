from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib
from src.config import MODEL_PATH

def train_demand_model(X, y):
    # 1. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train model
    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # 3. Evaluate
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    r2   = r2_score(y_test, y_pred)

    print(f"\n📊 Demand Model Evaluation")
    print(f"   MAE  : {mae:.2f} rides")
    print(f"   RMSE : {rmse:.2f} rides")
    print(f"   R²   : {r2:.4f}")

    # 4. Sanity check
    if r2 < 0.5:
        print(f"⚠️  WARNING: R² is low ({r2:.2f}) — demand features may be weak")
    else:
        print(f"✅ Demand model looks good!")

    # 5. Save
    joblib.dump(model, MODEL_PATH + "demand_model.pkl")
    print(f"💾 Demand model saved at {MODEL_PATH}demand_model.pkl")

    return model, X_test, y_test, y_pred