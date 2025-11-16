import pandas as pd
import numpy as np
from unidecode import unidecode

def _norm_cat(s: pd.Series) -> pd.Series:
    return (s.astype(str)
              .map(lambda x: unidecode(x))
              .str.lower()
              .str.replace(" ", "_", regex=False))

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).drop_duplicates()

    # Expected columns in this dataset: price, house_type, house_type_2, rooms, m2, elevator, garage, neighborhood, district
    # ---- NA & types
    if "house_type_2" in df.columns:
        df["house_type_2"] = df["house_type_2"].fillna("unknown")

    # coerce numerics
    for c in ["rooms", "m2", "price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # essential fields present
    need = [c for c in ["price", "m2", "rooms"] if c in df.columns]
    if need:
        df = df.dropna(subset=need)

    # m2 fix: tiny values (like 1–9) → *1000
    if "m2" in df.columns:
        df.loc[df["m2"] < 10, "m2"] = df.loc[df["m2"] < 10, "m2"] * 1000

    # normalize text columns
    for c in ["house_type", "house_type_2", "neighborhood", "district"]:
        if c in df.columns:
            df[c] = _norm_cat(df[c])

    # booleans to ints for elevator/garage if present
    for c in ["elevator", "garage"]:
        if c in df.columns:
            if df[c].dtype == bool:
                df[c] = df[c].astype(int)
            else:
                df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0})
                df[c] = df[c].fillna(0).astype(int)

    # outlier trimming
    if "m2" in df.columns:
        df = df[df["m2"] <= df["m2"].quantile(0.99)]
    if "price" in df.columns:
        df = df[df["price"] <= df["price"].quantile(0.99)]

    # target
    if "price" in df.columns:
        df["log_price"] = np.log1p(df["price"])

    return df

