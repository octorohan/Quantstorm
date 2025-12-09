# src/train.py
# Lightweight baseline trainer using sklearn (RandomForestRegressor).
# Train on concatenated per-ticker features for speed.

import pandas as pd
import numpy as np
import os
from src.features import add_basic_features, compute_30min_realized_vol
from src.dataset import create_windows, flatten_X, save_scaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

RAW_DIR = "data/raw"
MODEL_DIR = "models"

def build_features_for_ticker(csv_path):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_convert('America/New_York')
    df = add_basic_features(df)
    rv = compute_30min_realized_vol(df)
    # join rv back to df (forward-fill)
    merged = df.join(rv, how='left')
    merged['rv_30min'] = merged['rv_30min'].fillna(method='ffill')
    # drop early rows without rv
    merged = merged.dropna(subset=['rv_30min'])
    return merged

def load_all_and_concat(raw_dir=RAW_DIR):
    frames = []
    for fname in os.listdir(raw_dir):
        if fname.endswith("_1min.csv"):
            path = os.path.join(raw_dir, fname)
            try:
                frames.append(build_features_for_ticker(path))
            except Exception as e:
                print("Skipping", path, "due to", e)
    if not frames:
        raise RuntimeError("No data frames loaded. Check raw CSVs.")
    big = pd.concat(frames).sort_index()
    return big

def train_baseline():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_all_and_concat()
    # choose simple features
    features = ['Open','High','Low','Close','Volume','hl_spread','volume_imbalance']
    df = df.dropna(subset=features + ['rv_30min'])
    X_df = df[features]
    y = df['rv_30min']
    # Standardize
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_df)
    save_scaler(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    # quick train - RandomForest
    print("Training RandomForest baseline ...")
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(Xs, y.values)
    joblib.dump(model, os.path.join(MODEL_DIR, "baseline_model.pkl"))
    print("Saved baseline_model.pkl and scaler.pkl in models/")

if __name__ == "__main__":
    train_baseline()
