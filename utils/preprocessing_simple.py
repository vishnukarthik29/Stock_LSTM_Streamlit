from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np

api_key = "70CILADGUDA5YE75"

def load_data(symbol):
    """Simple data loading with basic technical indicators"""
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='full')
        data = data.sort_index()
        
        # Rename columns for easier access
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Clean data - remove any rows with missing values
        data = data.dropna()
        
        # Add basic technical indicators
        data = add_basic_features(data)
        
        # Fill any remaining NaN values
        data = data.ffill().bfill()
        
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data for {symbol}: {e}")

def add_basic_features(df):
    """Add basic technical and statistical features"""
    
    # Price-based features
    df['HL_pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['CO_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
    
    # Moving averages of different periods
    for window in [5, 10, 20, 50]:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}_ratio'] = df['Close'] / df[f'MA_{window}']
    
    # Exponential moving averages
    for window in [12, 26]:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_window = 20
    df['BB_middle'] = df['Close'].rolling(window=bb_window).mean()
    bb_std = df['Close'].rolling(window=bb_window).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
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
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    # Williams %R
    df['williams_r'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Day of week and month effects
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    
    return df

def prepare_features(df, target_col='Close'):
    """Select and prepare features for modeling"""
    
    # Define feature categories
    price_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Technical indicator features
    technical_features = [
        'HL_pct', 'CO_pct', 'MA_5', 'MA_10', 'MA_20', 'MA_50',
        'MA_5_ratio', 'MA_10_ratio', 'MA_20_ratio', 'MA_50_ratio',
        'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_histogram',
        'RSI', 'BB_middle', 'BB_upper', 'BB_lower', 'BB_width', 'BB_position',
        'volatility_10', 'volatility_30', 'price_position_14', 'price_position_30',
        'momentum_5', 'momentum_10', 'momentum_20', 'volume_ratio',
        'stoch_k', 'stoch_d', 'williams_r', 'ATR',
        'day_of_week', 'month', 'day_of_month'
    ]
    
    # Combine all features
    all_features = price_features + technical_features
    
    # Select features that exist in the dataframe
    available_features = [col for col in all_features if col in df.columns]
    
    # Ensure Close is always included
    if 'Close' not in available_features:
        available_features.append('Close')
    
    feature_df = df[available_features].copy()
    
    # Remove highly correlated features but preserve Close column
    feature_df = remove_highly_correlated_features(feature_df, threshold=0.95, preserve_cols=['Close'])
    
    # Final safety check - ensure Close column exists
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
        if column in preserve_cols:
            continue
        if any(upper_tri[column] > threshold):
            to_drop.append(column)

    return df.drop(columns=to_drop)
