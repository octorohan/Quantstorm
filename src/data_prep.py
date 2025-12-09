# src/data_prep.py
# Download 1-minute OHLC data for given tickers (period=7d), convert to America/New_York tz,
# filter to last 5 trading days and market hours (09:30-16:00 ET), and save per-ticker CSVs.

import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
import os

TICKERS = ["AAPL","META","AMZN","GOOGL","NFLX"]
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def download_and_save(ticker, period="7d", interval="1m", raw_dir=RAW_DIR):
    print(f"Downloading {ticker} {period} {interval} ...")
    # Explicitly set auto_adjust=False to avoid warning and keep raw OHLC
    df = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=False)
    if df is None or df.empty:
        print(f"Warning: no data for {ticker}")
        return None

    # Reset index so timestamp is a column we can work with reliably
    df = df.reset_index()
    # yfinance names the datetime column either 'Datetime' or 'index' depending on version
    dt_col = None
    for c in ["Datetime", "datetime", "index"]:
        if c in df.columns:
            dt_col = c
            break
    if dt_col is None:
        # fallback: assume first column is timestamp
        dt_col = df.columns[0]

    # Ensure datetime dtype and timezone
    df[dt_col] = pd.to_datetime(df[dt_col])
    if df[dt_col].dt.tz is None:
        df[dt_col] = df[dt_col].dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        df[dt_col] = df[dt_col].dt.tz_convert("America/New_York")

    # Use only the last 5 unique trading days (by date)
    unique_days = pd.Series(df[dt_col].dt.date).unique().tolist()
    unique_days = sorted(unique_days)
    if len(unique_days) > 5:
        keep_days = set(unique_days[-5:])
        mask = pd.Series(df[dt_col].dt.date).isin(keep_days).values
        df = df.loc[mask].copy()

    # set datetime index and sort
    df = df.set_index(dt_col).sort_index()

    # Filter to regular market hours 09:30-16:00 ET
    df = df.between_time("09:30", "16:00")

    # Save to CSV
    outpath = os.path.join(raw_dir, f"{ticker}_1min.csv")
    df.to_csv(outpath, index=True)
    print(f"Saved {outpath} rows={len(df)}")
    return outpath

def download_all(tickers=TICKERS):
    for t in tickers:
        try:
            download_and_save(t)
        except Exception as e:
            print(f"Error downloading {t}: {e}")

if __name__ == "__main__":
    download_all()
