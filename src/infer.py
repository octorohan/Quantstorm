# src/infer.py
# Simple inference wrapper using baseline_model.pkl

import joblib
import pandas as pd
import numpy as np
import os
from src.features import add_basic_features
MODEL_DIR = "models"

def infer_csv(csv_path, model_path=os.path.join(MODEL_DIR,"baseline_model.pkl"), scaler_path=os.path.join(MODEL_DIR,"scaler.pkl")):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if df.index.tzinfo is None:
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')
    df = add_basic_features(df)
    features = ['Open','High','Low','Close','Volume','hl_spread','volume_imbalance']
    X = df[features].ffill().fillna(0)
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    df['predicted_rv'] = np.nan
    df.loc[X.index, 'predicted_rv'] = preds
    return df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py path/to/ticker_1min.csv")
        sys.exit(1)
    out = infer_csv(sys.argv[1])
    print(out[['Close','predicted_rv']].tail())
