# src/clean_raw.py
# Small utility to clean the raw CSVs in data/raw/ that may contain extra header rows
# produced by some yfinance versions or manual edits.
#
# It will:
# - read each CSV with index_col=0, parse_dates=True
# - drop rows where 'Open' is not numeric
# - ensure index is timezone-aware (America/New_York)
# - save cleaned CSV back (overwrite)

import os
import glob
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

def clean_file(path):
    print("Cleaning", path)
    try:
        # read CSV (first column is assumed to be the datetime/index)
        df = pd.read_csv(path, header=0, index_col=0)
    except Exception as e:
        print("  read error:", e)
        return False

    # if the DataFrame has column names like ['Price','Adj Close','Close', ...] but first rows contain
    # ticker names, drop rows where 'Open' is non-numeric
    if 'Open' not in df.columns:
        # try to find a plausible Open-like column (case-insensitive)
        col_candidates = [c for c in df.columns if c.lower() in ('open','o')]
        if col_candidates:
            df.rename(columns={col_candidates[0]: 'Open'}, inplace=True)

    # Coerce Open to numeric; drop rows where it's NaN after coercion
    if 'Open' in df.columns:
        df['Open_num'] = pd.to_numeric(df['Open'], errors='coerce')
        before = len(df)
        df = df[df['Open_num'].notna()].copy()
        df.drop(columns=['Open_num'], inplace=True)
        after = len(df)
        print(f"  dropped {before-after} non-numeric rows")
    else:
        # If no Open column, attempt to infer by position (last resort)
        print("  Warning: 'Open' column not found, attempting positional inference")
        if df.shape[1] >= 5:
            df.columns = ['Open','High','Low','Close','Adj Close','Volume'][:df.shape[1]]
            df['Open_num'] = pd.to_numeric(df['Open'], errors='coerce')
            df = df[df['Open_num'].notna()].copy()
            df.drop(columns=['Open_num'], inplace=True)
        else:
            print("  Too few columns — skipping file")
            return False

    # Ensure index is datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        print("  index to_datetime failed:", e)
        return False

    # If timezone-naive, localize to UTC then convert to America/New_York
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
    else:
        try:
            df.index = df.index.tz_convert('America/New_York')
        except Exception:
            pass

    # Save back
    df.to_csv(path, index=True)
    print("  saved cleaned file, rows=", len(df))
    return True

def clean_all():
    pattern = os.path.join(RAW_DIR, "*_1min.csv")
    files = glob.glob(pattern)
    if not files:
        print("No files found at", pattern)
        return
    for p in files:
        clean_file(p)

if __name__ == "__main__":
    clean_all()
