# src/dataset.py
# Sliding-window dataset builder for supervised regression.
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

def create_windows(feature_df, target_series, lookback=60, stride=1):
    """
    feature_df: DataFrame indexed by timestamp, shape (T, F)
    target_series: Series indexed by timestamp (aligned to feature_df indices or coarser)
    lookback: how many minutes to use as encoder length (e.g., 60)
    Returns X (N, lookback, F), y (N,)
    """
    X, y = [], []
    idxs = feature_df.index
    for i in range(lookback, len(idxs)):
        window_idx = idxs[i - lookback:i]
        X.append(feature_df.loc[window_idx].values)
        # target aligned to the current time (i)
        t = idxs[i]
        if t in target_series.index:
            y.append(target_series.loc[t])
        else:
            # nearest forward/backward lookup
            nearest = target_series.index.asof(t)
            if pd.isna(nearest):
                y.append(np.nan)
            else:
                y.append(target_series.loc[nearest])
    X = np.array(X)
    y = np.array(y)
    # drop any samples with nan target
    mask = ~np.isnan(y)
    return X[mask], y[mask]

def flatten_X(X):
    """
    Flatten (N, seq, F) -> (N, seq*F) for sklearn models
    """
    N, seq, F = X.shape
    return X.reshape(N, seq * F)

def save_scaler(scaler, path):
    joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)
