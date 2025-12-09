# streamlit_app/app.py
# Minimal Streamlit app to upload a CSV and show candlestick + predicted volatility.
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.infer import infer_csv
import os

st.set_page_config(page_title="QuantStorm - Baseline", layout="wide")
st.title("QuantStorm — Baseline Volatility Predictor")

uploaded = st.file_uploader("Upload a 1-minute OHLC CSV (from data/raw)", type=["csv"])
if uploaded:
    # Save to temp
    tmp_path = os.path.join("data", "raw", "uploaded_tmp.csv")
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.info("Saved upload to data/raw/uploaded_tmp.csv — running baseline inference...")
    df = infer_csv(tmp_path)
    st.write("Preview")
    st.dataframe(df.tail(50))

    # Candlestick
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])])
    st.plotly_chart(fig, use_container_width=True)

    # Plot predicted vol
    if 'predicted_rv' in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df['predicted_rv'], name='predicted_rv'))
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Upload a ticker CSV (e.g., data/raw/AAPL_1min.csv) to run baseline predictions.")
