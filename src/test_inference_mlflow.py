# src/test_inference_mlflow.py
import mlflow
import pandas as pd
import numpy as np

# Must point to the SAME backend you used
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ⬇️ paste your run_id here (from the UI URL)
RUN_ID = "1012c56e5cfa444c9b60add8ee007ae7"

model_uri = f"runs:/{RUN_ID}/model_pipeline"

# Load either with sklearn flavor…
sk_model = mlflow.sklearn.load_model(model_uri)

# …or with the generic pyfunc flavor (both work)
# sk_model = mlflow.pyfunc.load_model(model_uri)

# Minimal example row (use your dataset’s columns; others are optional-safe due to preprocessing)
sample = pd.DataFrame([{
    "m2": 85,
    "rooms": 3,
    "elevator": 1,
    "garage": 0,
    "house_type": "piso",
    "house_type_2": "reformado",
    "neighborhood": "sol",
    "district": "centro"
}])

log_price = float(sk_model.predict(sample)[0])
price = float(np.expm1(log_price))  # inverse of log1p
print({"log_price": log_price, "price": price})
