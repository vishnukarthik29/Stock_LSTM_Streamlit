import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime, timedelta

from alpha_vantage.timeseries import TimeSeries

# 🔑 Enter your Alpha Vantage API key here
api_key = '70CILADGUDA5YE75'  # Replace with your actual key

# 🌐 Load stock data using Alpha Vantage
@st.cache_data(show_spinner=False)
def load_data(symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()
        df = data[['4. close']]
        df.columns = ['Close']
        return df
    except Exception as e:
        st.error(f"❌ Could not load data for symbol {symbol}: {e}")
        return pd.DataFrame()

# 🔄 Create sequences for LSTM
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# 📱 Streamlit UI
st.set_page_config(page_title="📈 Stock Price Prediction (LSTM)", layout="centered")
st.title("📊 Indian Stock Price Prediction with LSTM")
st.write("Using Alpha Vantage + LSTM + Streamlit")

stocks = ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE', 'WIPRO.BSE']
stock = st.selectbox("Choose a Stock", stocks)

if st.button("🔍 Predict"):
    df = load_data(stock)

    if df.empty:
        st.warning("No data found.")
        st.stop()

    st.subheader("📄 Recent Stock Data")
    st.dataframe(df.tail())
    st.line_chart(df['Close'])

    # Preprocessing
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Train-Test split
    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    with st.spinner("⏳ Training LSTM model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot results
    st.subheader("📉 Predicted vs Actual Prices")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label="Actual")
    ax.plot(predictions, label="Predicted")
    ax.legend()
    st.pyplot(fig)

    # Predict next day
    last_60_days = scaled_data[-60:]
    next_input = last_60_days.reshape(1, 60, 1)
    next_price = model.predict(next_input)
    next_price = scaler.inverse_transform(next_price)[0][0]

    st.success(f"📌 Predicted Next Day Closing Price: ₹{next_price:.2f}")

    st.info("Note: This is a demo app and not financial advice.")
