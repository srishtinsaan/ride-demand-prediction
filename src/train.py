from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np
from src.config import MODEL_PATH

def train_model(X, y, model_name):
    # 1. Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train model
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
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

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    print(f"\n📊 Model: {model_name}")
    print(f"   MAE  : ${mae:.2f}")
    print(f"   RMSE : ${rmse:.2f}")
    print(f"   R²   : {r2:.4f}")

    # 4. Sanity check
    if r2 < 0.5:
        print(f"⚠️  WARNING: R² is low ({r2:.2f}) — check your features or data cleaning")
    else:
        print(f"✅ Model looks good!")

    # 5. Save model
    joblib.dump(model, MODEL_PATH + model_name)
    print(f"💾 Model saved at {MODEL_PATH + model_name}")

    return model, X_test, y_test, y_pred