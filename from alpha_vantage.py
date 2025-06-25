from alpha_vantage.timeseries import TimeSeries
import pandas as pd

api_key = "70CILADGUDA5YE75"

def load_data(symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()
        df = data[['4. close']]
        df.columns = ['Close']
        return df
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}")
    
the above code preprocessing.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from utils.preprocessing import load_data
import datetime

# Page config
st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")
st.title("📊 Stock Price Prediction using LSTM")

# Sidebar
stock = st.selectbox("Select Stock", ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE', 'WIPRO.BSE'])
n_days = st.slider("Prediction Horizon (days)", min_value=30, max_value=365, value=60)

if st.button("Predict"):
    # Load data
    df = load_data(stock)
    st.subheader("🔍 Basic Data Inspection")
    st.dataframe(df.tail())
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:")
    st.write(df.dtypes)

    # EDA
    st.subheader("📈 EDA - Univariate Analysis")
    fig, ax = plt.subplots()
    ax.plot(df['Close'], label="Close Price")
    ax.set_title(f"{stock} Closing Price Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    st.subheader("📊 EDA - Descriptive Statistics")
    st.write(df.describe())

    # Data Transformation
    st.subheader("🔧 Data Transformation")
    df['Close_diff'] = df['Close'].diff()
    st.write("Outlier check (diff):")
    st.write(df['Close_diff'].describe())

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['Close']])
    st.write("📉 After Scaling:")
    st.line_chart(scaled_data)

    # Prepare training data
    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - n_days):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i:i+n_days, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    st.write("✅ X_train shape:", X_train.shape)
    st.write("✅ y_train shape:", y_train.shape)

    # Build LSTM Model
    st.subheader("🧠 Building LSTM Model")
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=n_days))
    model.compile(optimizer='adam', loss='mean_squared_error')
    with st.spinner("⏳ Training LSTM model..."):
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    # model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Predict
    st.subheader("📡 Predicting Future Stock Prices")
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(np.reshape(y_pred_scaled[:, -1:], (-1, 1)))
    y_actual = scaler.inverse_transform(np.reshape(y_test[:, -1:], (-1, 1)))

    # Plot prediction
    fig2, ax2 = plt.subplots()
    ax2.plot(y_actual, color="blue", label="Actual Price")
    ax2.plot(y_pred, color="red", label="Predicted Price")
    ax2.set_title(f"{stock} Prediction vs Actual")
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    # Evaluation
    st.subheader("📏 Model Evaluation")
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    st.write("✅ Mean Squared Error:", mse)
    st.write("✅ Mean Absolute Error:", mae)
    # Predict next day
    last_60_days = scaled_data[-60:]
    next_input = last_60_days.reshape(1, 60, 1)
    next_price = model.predict(next_input)
    next_price = scaler.inverse_transform(next_price)[0][0]
    st.success(f"📌 Predicted Next Day Closing Price: ₹{next_price:.2f}")

    # Recommendations
    st.subheader("💡 Insights & Recommendations")
    if y_pred[-1] > y_actual[-1]:
        st.success("The model suggests a possible upward trend in the coming days.")
    else:
        st.warning("The model predicts a potential downward or stagnant trend.")

the above code is app.py