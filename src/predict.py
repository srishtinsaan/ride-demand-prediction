import joblib
import pandas as pd
from src.config import MODEL_PATH

def load_model(model_name):
    return joblib.load(MODEL_PATH + model_name)

def predict(model, input_dict):
    df = pd.DataFrame([input_dict])
    return model.predict(df)[0]