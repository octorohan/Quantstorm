# src/dataset_windowed.py
"""
Create sliding-window datasets for LSTM training.
Reads all *_1min.csv files in data/raw/, computes features (using src.features),
computes 30-min realized volatility (target), aligns target, and creates
windows of shape (N, lookback, F) with target y (aligned to window end).
Saves train/val/test as .npz in data/processed/.
This version builds windows per-ticker to avoid mixing tickers and to ensure
every window has identical shape.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features import add_basic_features, compute_30min_realized_vol

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

DEFAULT_FEATURES = ["Open", "High", "Low", "Close", "Volume", "hl_spread", "volume_imbalance"]

def build_per_ticker_frames():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*_1min.csv")))
    frames = {}
    for p in files:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        # ensure tz-aware index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")
        df = add_basic_features(df)
        rv = compute_30min_realized_vol(df)
        merged = df.join(rv, how="left")
        merged["rv_30min"] = merged["rv_30min"].ffill()
        merged = merged.dropna(subset=DEFAULT_FEATURES + ["rv_30min"])
        # ticker name from filename
        ticker = os.path.basename(p).split("_")[0]
        merged["ticker"] = ticker
        frames[ticker] = merged
    if not frames:
        raise RuntimeError("No raw CSVs found in data/raw/")
    return frames

def create_windows_per_ticker(frame, lookback=60):
    """
    frame: DataFrame for a single ticker, indexed by minute (sorted)
    returns X (n_windows, lookback, F) and y (n_windows,)
    """
    features = DEFAULT_FEATURES
    # ensure sorted by time
    frame = frame.sort_index()
    n = len(frame)
    if n <= lookback:
        return np.empty((0, lookback, len(features)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    X_list = []
    y_list = []
    # use iloc to ensure exact lookback rows per window
    for i in range(lookback, n):
        window = frame.iloc[i - lookback:i][features].values.astype(np.float32)  # shape (lookback, F)
        target = frame.iloc[i]["rv_30min"]
        if np.isnan(target):
            continue
        X_list.append(window)
        y_list.append(np.float32(target))
    if not X_list:
        return np.empty((0, lookback, len(features)), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.stack(X_list), np.array(y_list, dtype=np.float32)

def build_and_save_all(lookback=60):
    frames = build_per_ticker_frames()
    Xs = []
    ys = []
    for ticker, df in frames.items():
        X_t, y_t = create_windows_per_ticker(df, lookback=lookback)
        if len(X_t):
            Xs.append(X_t)
            ys.append(y_t)
        print(f"{ticker}: windows={len(X_t)}")
    if not Xs:
        raise RuntimeError("No windows created from any ticker.")
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    print("Combined windows shape:", X_all.shape, y_all.shape)
    split_and_save(X_all, y_all, PROCESSED_DIR)

def split_and_save(X, y, outpath, val_frac=0.1, test_frac=0.1, seed=42):
    n = len(X)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    X = X[idx]; y = y[idx]
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_val - n_test
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    np.savez_compressed(os.path.join(outpath, "windows_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(outpath, "windows_val.npz"), X=X_val, y=y_val)
    np.savez_compressed(os.path.join(outpath, "windows_test.npz"), X=X_test, y=y_test)
    print(f"Saved train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

if __name__ == "__main__":
    build_and_save_all(lookback=60)
