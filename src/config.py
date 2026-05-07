import os

# Base directory = project root (wherever config.py lives, go 2 levels up)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths built relative to project root
DATA_PATH   = os.path.join(BASE_DIR, "data", "raw") + os.sep
MODEL_PATH  = os.path.join(BASE_DIR, "models") + os.sep
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs") + os.sep

# Auto-create directories if they don't exist
for path in [DATA_PATH, MODEL_PATH, OUTPUT_PATH]:
    os.makedirs(path, exist_ok=True)