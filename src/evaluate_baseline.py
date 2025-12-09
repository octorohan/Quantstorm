# src/evaluate_baseline.py
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from src.infer import infer_csv
from src.features import compute_30min_realized_vol
import matplotlib.pyplot as plt
import os

def evaluate_one(csv_path):
    df_pred = infer_csv(csv_path)  # returns minute-level df with 'predicted_rv'
    # compute 30-min realized vol (aligned to window end)
    rv30 = compute_30min_realized_vol(pd.read_csv(csv_path, index_col=0, parse_dates=True))
    # align: reindex predicted to rv30.index using asof (nearest)
    df_pred.index = pd.to_datetime(df_pred.index)
    preds = []
    reals = []
    for t in rv30.index:
        # take last predicted value at or before t
        try:
            p = df_pred.loc[:t]['predicted_rv'].iloc[-1]
            r = rv30.loc[t, 'rv_30min']
            preds.append(p)
            reals.append(r)
        except Exception:
            continue
    preds = np.array(preds)
    reals = np.array(reals)
    mask = ~np.isnan(preds) & ~np.isnan(reals)
    if mask.sum() == 0:
        print("No aligned samples")
        return
    rmse = np.sqrt(mean_squared_error(reals[mask], preds[mask]))
    print(f"RMSE (aligned 30-min): {rmse:.6f} over {mask.sum()} samples")
    # simple plot
    plt.figure(figsize=(10,4))
    plt.plot(reals[mask], label='real_30min')
    plt.plot(preds[mask], label='pred_30min')
    plt.legend()
    plt.title(os.path.basename(csv_path))
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate_baseline.py data/raw/AAPL_1min.csv")
    else:
        evaluate_one(sys.argv[1])
