# utils/preprocessing.py

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load historical data
# def load_data(stock):
#     import yfinance as yf
#     df = yf.download(stock, start='2010-01-01')
#     print(f"Downloaded data shape for {stock}: {df.shape}")
#     if df.empty:
#         raise ValueError(f"No data found for symbol: {stock}")
#     df = df[['Close']]
#     return df
def load_data(stock: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Load stock price data for a given ticker using yfinance.
    Falls back to a default if data is unavailable.
    """
    df = yf.download(
        tickers=stock,
        period=period,
        interval=interval,
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data found for symbol: {stock} with period={period} and interval={interval}")

    # Ensure 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("No 'Close' column found in data!")

    return df[['Close']]
# from nsepy import get_history
# from datetime import date

# def load_data(symbol):
#     df = get_history(symbol=symbol,
#                      start=date(2010,1,1),
#                      end=date.today())
#     df = df[['Close']]
#     return df

# Preprocess for LSTM
def preprocess_data(df, time_step=60):
    if df.empty or len(df) < time_step:
        raise ValueError(f"Insufficient data for preprocessing. Data length: {len(df)}")

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(time_step, len(df_scaled)):
        X.append(df_scaled[i-time_step:i, 0])
        y.append(df_scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

# Build LSTM Model
def build_lstm(X, y):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model
