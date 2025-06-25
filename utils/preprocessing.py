# # utils/preprocessing.py

# import yfinance as yf
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout

# # Load historical data
# # def load_data(stock):
# #     import yfinance as yf
# #     df = yf.download(stock, start='2010-01-01')
# #     print(f"Downloaded data shape for {stock}: {df.shape}")
# #     if df.empty:
# #         raise ValueError(f"No data found for symbol: {stock}")
# #     df = df[['Close']]
# #     return df
# def load_data(stock: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
#     """
#     Load stock price data for a given ticker using yfinance.
#     Falls back to a default if data is unavailable.
#     """
#     df = yf.download(
#         tickers=stock,
#         period=period,
#         interval=interval,
#         auto_adjust=True,
#         threads=True,
#         progress=False,
#     )

#     if df.empty:
#         raise ValueError(f"No data found for symbol: {stock} with period={period} and interval={interval}")

#     # Ensure 'Close' column exists
#     if 'Close' not in df.columns:
#         raise ValueError("No 'Close' column found in data!")

#     return df[['Close']]
# # from nsepy import get_history
# # from datetime import date

# # def load_data(symbol):
# #     df = get_history(symbol=symbol,
# #                      start=date(2010,1,1),
# #                      end=date.today())
# #     df = df[['Close']]
# #     return df

# # from alpha_vantage.timeseries import TimeSeries
# # import pandas as pd

# # api_key = "YOUR_API_KEY"  # 🔑 Replace with your actual API key

# # def load_data(symbol):
# #     ts = TimeSeries(key=api_key, output_format='pandas')
    
# #     try:
# #         data, _ = ts.get_daily(symbol=symbol, outputsize='full')
# #         data = data.sort_index()  # Make sure it's in chronological order
# #         df = data[['4. close']]  # We'll use the 'close' column
# #         df.columns = ['Close']
# #         return df
# #     except Exception as e:
# #         raise ValueError(f"❌ Could not load data for symbol {symbol}. Reason: {e}")

# from alpha_vantage.timeseries import TimeSeries
# import pandas as pd

# api_key = "70CILADGUDA5YE75"

# def load_data(symbol):
#     ts = TimeSeries(key=api_key, output_format='pandas')
#     try:
#         data, _ = ts.get_daily(symbol=symbol, outputsize='full')
#         data = data.sort_index()
#         df = data[['4. close']]
#         df.columns = ['Close']
#         return df
#     except Exception as e:
#         raise ValueError(f"Error fetching data for {symbol}: {e}")


# # Preprocess for LSTM
# def preprocess_data(df, time_step=60):
#     if df.empty or len(df) < time_step:
#         raise ValueError(f"Insufficient data for preprocessing. Data length: {len(df)}")

#     scaler = MinMaxScaler()
#     df_scaled = scaler.fit_transform(df)

#     X, y = [], []
#     for i in range(time_step, len(df_scaled)):
#         X.append(df_scaled[i-time_step:i, 0])
#         y.append(df_scaled[i, 0])

#     X = np.array(X)
#     y = np.array(y)
#     X = X.reshape(X.shape[0], X.shape[1], 1)
#     return X, y, scaler

# # Build LSTM Model
# def build_lstm(X, y):
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50))
#     model.add(Dropout(0.2))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     model.fit(X, y, epochs=5, batch_size=32, verbose=0)
#     return model










from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from ta.utils import dropna

api_key = "70CILADGUDA5YE75"

def load_data(symbol):
    """Enhanced data loading with technical indicators"""
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()
        
        # Rename columns for easier access
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Clean data - remove any rows with missing values
        data = dropna(data)
        
        # Add technical indicators (with error handling)
        try:
            data = add_all_ta_features(
                data, open="Open", high="High", low="Low", close="Close", volume="Volume"
            )
        except Exception as e:
            print(f"Warning: Could not add all technical indicators: {e}")
            # Continue with basic indicators
            pass
        
        # Add custom features
        data = add_custom_features(data)
        
        # Fill any remaining NaN values
        data = data.ffill().bfill()
        
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}")

def add_custom_features(df):
    """Add custom technical and statistical features"""
    
    # Price-based features
    df['HL_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['CO_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Moving averages of different periods
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
    
    # Volatility features
    df['volatility_10'] = df['Close'].rolling(window=10).std()
    df['volatility_30'] = df['Close'].rolling(window=30).std()
    
    # Price position within recent range
    for window in [14, 30]:
        df[f'price_position_{window}'] = (
            (df['Close'] - df['Low'].rolling(window).min()) / 
            (df['High'].rolling(window).max() - df['Low'].rolling(window).min())
        )
    
    # Momentum features
    df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
    df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
    df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
    
    # Volume features
    df['volume_ma_10'] = df['Volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma_10']
    
    # Day of week and month effects
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    
    return df

def prepare_features(df, target_col='Close'):
    """Select and prepare features for modeling"""
    
    # Define feature categories
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Technical indicator features (selecting most important ones)
    ta_features = [
        'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_mfi',
        'volume_em', 'volume_sma_em', 'volume_vpt', 'volume_nvi', 'volume_vwap',
        'volatility_bbm', 'volatility_bbh', 'volatility_bbl', 'volatility_bbw',
        'volatility_atr', 'volatility_kcm', 'volatility_kch', 'volatility_kcl',
        'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
        'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow', 'trend_adx',
        'trend_vortex_vi_pos', 'trend_vortex_vi_neg', 'trend_trix',
        'momentum_rsi', 'momentum_stoch_rsi', 'momentum_tsi', 'momentum_uo',
        'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr', 'momentum_ao',
        'momentum_kama', 'momentum_roc', 'momentum_ppo', 'momentum_pvo'
    ]
    
    # Custom features
    custom_features = [
        'HL_pct', 'CO_pct', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
        'MA_5_ratio', 'MA_10_ratio', 'MA_20_ratio', 'MA_50_ratio',
        'volatility_10', 'volatility_30', 'price_position_14', 'price_position_30',
        'momentum_5', 'momentum_10', 'momentum_20', 'volume_ratio',
        'day_of_week', 'month', 'day_of_month'
    ]
    
    # Ensure Close column is always included
    if 'Close' not in df.columns:
        raise ValueError("Close column not found in dataframe")
    
    # Combine all features
    all_features = price_features + custom_features
    
    # Add technical indicators that exist in the dataframe
    existing_ta_features = [col for col in ta_features if col in df.columns]
    all_features.extend(existing_ta_features)
    
    # Select features that exist in the dataframe
    available_features = [col for col in all_features if col in df.columns]
    
    # Ensure Close is in available features
    if 'Close' not in available_features:
        available_features.append('Close')
    
    feature_df = df[available_features].copy()
    
    # Remove highly correlated features but preserve Close column
    feature_df = remove_highly_correlated_features(feature_df, threshold=0.95, preserve_cols=['Close'])
    
    # Final safety check
    if 'Close' not in feature_df.columns:
        feature_df['Close'] = df['Close']
    
    return feature_df

def remove_highly_correlated_features(df, threshold=0.95, preserve_cols=['Close']):
    """Remove highly correlated features while preserving important columns"""
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = []
    for column in upper_tri.columns:
        if column not in preserve_cols and any(upper_tri[column] > threshold):
            to_drop.append(column)
    
    return df.drop(columns=to_drop)