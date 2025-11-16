# src/train_model.py
import os
import joblib
import numpy as np
import pandas as pd
import sys

# Use an alias to avoid any shadowing issues
import mlflow as mlf
import mlflow.sklearn as mlf_sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from data_cleaning import load_and_clean_data

# Point MLflow to local SQLite DB (no deprecation warnings)
mlf.set_tracking_uri("sqlite:///mlflow.db")

# CI-safe experiment init: allow CI to use a different name, and ensure local artifact path
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Madrid_House_Prices")
exp = mlf.get_experiment_by_name(EXP_NAME)
if exp is None:
    exp_id = mlf.create_experiment(
        EXP_NAME,
        artifact_location="file:./mlruns"  # relative; valid on Linux & Windows
    )
    mlf.set_experiment(experiment_id=exp_id)
else:
    mlf.set_experiment(EXP_NAME)


def train_model(csv_path: str):
    # 1) Load & clean data
    df = load_and_clean_data(csv_path)

    # 2) Ensure target exists
    if "log_price" not in df.columns:
        if "price" not in df.columns:
            raise ValueError("Neither 'log_price' nor 'price' found in the dataset.")
        df["log_price"] = np.log1p(df["price"])

    target = "log_price"
    y = df[target]
    X = df.drop(columns=[target, "price"], errors="ignore")

    # 3) Candidate features (use only those present)
    numeric_features_all = ["m2", "rooms", "elevator", "garage"]
    categorical_features_all = ["house_type", "house_type_2", "neighborhood", "district"]

    num_used = [c for c in numeric_features_all if c in X.columns]
    cat_used = [c for c in categorical_features_all if c in X.columns]

    if not num_used and not cat_used:
        raise ValueError(
            f"No usable features found. Available columns: {list(X.columns)}\n"
            f"Expected any of: {numeric_features_all + categorical_features_all}"
        )

    # 4) Preprocessing
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_used),
        ("cat", categorical_transformer, cat_used)
    ])

    # 5) Model pipeline
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # 6) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7) Hyperparameter search
    param_grid = {
        "model__n_estimators": [200, 300, 400],
        "model__max_depth": [15, 20, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # 8) Train + log (single run block)
    with mlf.start_run():
        search.fit(X_train, y_train)

        preds = search.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        # Log params/metrics
        mlf.log_params(search.best_params_)
        mlf.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

        # Save local artifact (pipeline includes preprocessing)
        os.makedirs("models", exist_ok=True)
        best_pipe = search.best_estimator_
        joblib.dump(best_pipe, "models/model.pkl")

        # Log to MLflow with signature + example (no deprecation warnings)
        input_example = X_test.head(2).copy()
        # Avoid MLflow int/NaN warning by casting int-like to float if present
        for c in ["m2", "rooms", "elevator", "garage"]:
            if c in input_example.columns:
                input_example[c] = input_example[c].astype(float)

        sig = infer_signature(input_example, preds[:2])

        mlf_sklearn.log_model(
            best_pipe,
            name="model_pipeline",
            input_example=input_example,
            signature=sig
        )

        print("✅ Training complete")
        print(f"   RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")
        print(f"   Features used: numeric={num_used}, categorical={cat_used}")
        print("   Saved: models/model.pkl")

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "numeric_used": num_used, "categorical_used": cat_used}


if __name__ == "__main__":
    # If a path is provided, use it; else prefer the sample if present
    sample = os.path.join("data", "madrid_sample.csv")
    full   = os.path.join("data", "house_price_madrid_14_08_2022.csv")
    csv_path = sys.argv[1] if len(sys.argv) > 1 else (sample if os.path.exists(sample) else full)
    metrics = train_model(csv_path)
    print(metrics)



