# src/features.py
import pandas as pd
import numpy as np

def add_basic_features(df):
    df = df.copy()
    df['hl_spread'] = df['High'] - df['Low']
    # volume imbalance: (vol_t - vol_t-1) / (vol_t + vol_t-1)
    df['vol_prev'] = df['Volume'].shift(1).bfill()
    df['volume_imbalance'] = (df['Volume'] - df['vol_prev']) / (df['Volume'] + df['vol_prev'] + 1e-9)
    df.drop(columns=['vol_prev'], inplace=True)
    return df

def realized_volatility(series_returns):
    return np.sqrt(np.sum(np.square(series_returns.values)))

def compute_30min_realized_vol(df):
    df = df.copy()
    df['logret'] = np.log(df['Close']).diff().fillna(0)
    rv_list = []
    groups = df['logret'].resample('30T')
    for ts, grp in groups:
        if len(grp) == 0:
            rv_list.append((ts, np.nan))
        else:
            rv_list.append((ts, realized_volatility(grp)))
    rv_df = pd.DataFrame(rv_list, columns=['timestamp','rv_30min']).set_index('timestamp')
    return rv_df
