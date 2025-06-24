# app.py

import streamlit as st
from utils.preprocessing import load_data, preprocess_data, build_lstm
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")
st.title("📈 Indian Stock Price Predictor using LSTM")

stock = st.text_input("Enter NSE Stock Code (e.g., TCS.NS)", "TCS.NS")
st.sidebar.markdown("### Settings")
period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "10y", "max"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
if st.button("Predict"):
    try:
        df = load_data(stock, period=period, interval=interval)
        ...
       
        st.subheader("📊 Recent Stock Data")
        st.dataframe(df.tail())
        st.write("Loaded data shape:", df.shape)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    # Load and show raw data
    


    # Prepare and train model
    X, y, scaler = preprocess_data(df)
    model = build_lstm(X, y)

    # Predict
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform(y.reshape(-1, 1))

    # Chart
    st.subheader("📉 Actual vs Predicted Close Price")
    fig, ax = plt.subplots()
    ax.plot(y_actual, label="Actual")
    ax.plot(y_pred, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Model insights
    st.markdown("### 🔍 Model Summary")
    st.write("- LSTM trained on past 60 days to predict next day’s closing price.")
    st.write("- Predictions follow actual trends well in smoother patterns.")
    st.warning("⚠️ For education only – not financial advice.")
