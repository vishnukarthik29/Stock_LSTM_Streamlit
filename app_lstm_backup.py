import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from utils.preprocessing_simple import load_data, prepare_features
import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="📈 Advanced Stock Price Prediction", layout="wide")
st.title("📊 Advanced Stock Price Prediction using Enhanced LSTM")

# Sidebar
st.sidebar.header("🔧 Model Configuration")
stock = st.sidebar.selectbox("Select Stock", ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE', 'WIPRO.BSE'])
n_days = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=30, value=5)
sequence_length = st.sidebar.slider("Sequence Length", min_value=30, max_value=120, value=60)
n_features = st.sidebar.slider("Number of Features", min_value=5, max_value=50, value=20)

# Advanced options
st.sidebar.subheader("🧠 Advanced Options")
epochs = st.sidebar.slider("Training Epochs", min_value=20, max_value=200, value=100)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
lstm_units = st.sidebar.slider("LSTM Units", min_value=32, max_value=256, value=100)
dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2)
use_bidirectional = st.sidebar.checkbox("Use Bidirectional LSTM", value=True)

def create_sequences(data, target, sequence_length, n_days):
    """Create sequences for multi-step prediction"""
    X, y = [], []
    for i in range(sequence_length, len(data) - n_days + 1):
        X.append(data[i-sequence_length:i])
        y.append(target[i:i+n_days])
    return np.array(X), np.array(y)

def build_enhanced_model(input_shape, n_days, lstm_units, dropout_rate, use_bidirectional):
    """Build enhanced LSTM model with regularization"""
    model = Sequential()
    
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True, 
                                   kernel_regularizer=l2(0.001)), 
                              input_shape=input_shape))
    else:
        model.add(LSTM(units=lstm_units, return_sequences=True, 
                      kernel_regularizer=l2(0.001), input_shape=input_shape))
    
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units=lstm_units//2, return_sequences=True,
                                   kernel_regularizer=l2(0.001))))
    else:
        model.add(LSTM(units=lstm_units//2, return_sequences=True,
                      kernel_regularizer=l2(0.001)))
    
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    
    if use_bidirectional:
        model.add(Bidirectional(LSTM(units=lstm_units//4, kernel_regularizer=l2(0.001))))
    else:
        model.add(LSTM(units=lstm_units//4, kernel_regularizer=l2(0.001)))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(dropout_rate/2))
    model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(units=n_days))
    
    model.compile(optimizer=Adam(learning_rate=0.001), 
                 loss='huber', 
                 metrics=['mae'])
    
    return model

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

if st.button("🚀 Train and Predict", type="primary"):
    with st.spinner("📊 Loading and preprocessing data..."):
        # Load data with technical indicators
        df = load_data(stock)
        feature_df = prepare_features(df)
        
        # Debug information
        st.write("🔍 **Debug Information:**")
        st.write(f"Original data columns: {list(df.columns)}")
        st.write(f"Feature data columns: {list(feature_df.columns)}")
        st.write(f"Feature data shape: {feature_df.shape}")
        
        # Display data info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Raw Data Overview")
            st.write(f"Dataset shape: {df.shape}")
            st.write(f"Features available: {len(feature_df.columns)}")
            st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail())
        
        with col2:
            st.subheader("📊 Price Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df.index[-252:], df['Close'].iloc[-252:], label="Close Price")
            ax.set_title(f"{stock} - Last Year Price Movement")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (₹)")
            ax.legend()
            st.pyplot(fig)
    
    # Feature selection and preparation
    with st.spinner("🔧 Preparing features..."):
        # Ensure Close column exists
        if 'Close' not in feature_df.columns:
            st.error("❌ 'Close' column not found in feature data. Please check data preprocessing.")
            st.stop()
        
        # Select top features based on correlation with target
        correlations = feature_df.corr()['Close'].abs().sort_values(ascending=False)
        top_features = correlations.head(n_features).index.tolist()
        
        # Ensure Close is included
        if 'Close' not in top_features:
            top_features.append('Close')
        
        selected_features = feature_df[top_features].copy()
        
        st.subheader("🎯 Selected Features")
        st.write(f"Top {len(top_features)} features selected:")
        feature_importance = correlations[top_features].sort_values(ascending=False)
        st.bar_chart(feature_importance[1:])  # Exclude Close itself
    
    # Data preprocessing
    with st.spinner("⚙️ Preprocessing data..."):
        # Use RobustScaler for better handling of outliers
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Scale features
        scaled_features = feature_scaler.fit_transform(selected_features)
        scaled_target = target_scaler.fit_transform(df[['Close']])
        
        # Create sequences
        X, y = create_sequences(scaled_features, scaled_target.flatten(), sequence_length, n_days)
        
        st.write(f"✅ Created {len(X)} sequences")
        st.write(f"✅ Feature matrix shape: {X.shape}")
        st.write(f"✅ Target matrix shape: {y.shape}")
    
    # Train-test split (time series aware)
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Build and train model
    st.subheader("🧠 Training Enhanced LSTM Model")
    
    with st.spinner("🏗️ Building model architecture..."):
        model = build_enhanced_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            n_days=n_days,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            use_bidirectional=use_bidirectional
        )
        
        st.write("📋 Model Architecture:")
        st.text(f"Input Shape: {X_train.shape[1:2]}")
        st.text(f"LSTM Units: {lstm_units}")
        st.text(f"Bidirectional: {use_bidirectional}")
        st.text(f"Dropout Rate: {dropout_rate}")
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7)
    ]
    
    # Training
    with st.spinner(f"🎯 Training model for {epochs} epochs..."):
        progress_bar = st.progress(0)
        
        class ProgressCallback:
            def __init__(self, progress_bar):
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / epochs
                self.progress_bar.progress(progress)
        
        # Custom callback for progress
        progress_callback = ProgressCallback(progress_bar)
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
    
    # Training history visualization
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Training Loss")
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Training MAE")
        fig, ax = plt.subplots()
        ax.plot(history.history['mae'], label='Training MAE')
        ax.plot(history.history['val_mae'], label='Validation MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.legend()
        st.pyplot(fig)
    
    # Predictions
    st.subheader("🔮 Model Predictions")
    
    with st.spinner("📊 Generating predictions..."):
        # Predict on test set
        y_pred_scaled = model.predict(X_test)
        
        # Transform back to original scale
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_actual = target_scaler.inverse_transform(y_test.reshape(-1, n_days))
        
        # Calculate metrics for each prediction day
        metrics_by_day = {}
        for day in range(n_days):
            day_metrics = calculate_metrics(y_actual[:, day], y_pred[:, day])
            metrics_by_day[f'Day {day+1}'] = day_metrics
    
    # Visualize predictions
    st.subheader("📈 Prediction vs Actual")
    
    # Plot for each prediction day
    fig, axes = plt.subplots(min(n_days, 3), 1, figsize=(12, 4*min(n_days, 3)))
    if n_days == 1:
        axes = [axes]
    
    for i in range(min(n_days, 3)):
        axes[i].plot(y_actual[:, i], label=f'Actual Day {i+1}', alpha=0.7)
        axes[i].plot(y_pred[:, i], label=f'Predicted Day {i+1}', alpha=0.7)
        axes[i].set_title(f'{stock} - Day {i+1} Prediction')
        axes[i].set_xlabel('Test Samples')
        axes[i].set_ylabel('Price (₹)')
        axes[i].legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Metrics display
    st.subheader("📏 Model Performance Metrics")
    
    metrics_df = pd.DataFrame(metrics_by_day).T
    st.dataframe(metrics_df.round(4))
    
    # Overall performance
    overall_mape = np.mean([metrics_by_day[day]['MAPE'] for day in metrics_by_day.keys()])
    overall_r2 = np.mean([metrics_by_day[day]['R²'] for day in metrics_by_day.keys()])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall MAPE", f"{overall_mape:.2f}%")
    with col2:
        st.metric("Overall R² Score", f"{overall_r2:.4f}")
    with col3:
        accuracy = max(0, 100 - overall_mape)
        st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
    
    # Future prediction
    st.subheader("🔮 Future Price Prediction")
    
    # Use the last sequence to predict future prices
    last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, X.shape[2])
    future_pred_scaled = model.predict(last_sequence)
    future_pred = target_scaler.inverse_transform(future_pred_scaled)[0]
    
    current_price = df['Close'].iloc[-1]
    
    st.write("🎯 **Next Few Days Prediction:**")
    for i, price in enumerate(future_pred):
        change = ((price - current_price) / current_price) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"Day {i+1}:")
        with col2:
            st.write(f"₹{price:.2f}")
        with col3:
            color = "🟢" if change > 0 else "🔴" if change < 0 else "🟡"
            st.write(f"{color} {change:+.2f}%")
    
    # Insights and recommendations
    st.subheader("💡 AI Insights & Recommendations")
    
    avg_future_price = np.mean(future_pred)
    trend = "upward" if avg_future_price > current_price else "downward"
    confidence = min(95, max(60, accuracy))
    
    if overall_mape < 5:
        confidence_level = "High"
        confidence_color = "🟢"
    elif overall_mape < 10:
        confidence_level = "Medium"
        confidence_color = "🟡"
    else:
        confidence_level = "Low"
        confidence_color = "🔴"
    
    st.write(f"**Trend Analysis:** The model predicts a {trend} trend over the next {n_days} days.")
    st.write(f"**Confidence Level:** {confidence_color} {confidence_level} ({confidence:.1f}%)")
    st.write(f"**Model Performance:** MAPE of {overall_mape:.2f}% indicates {'excellent' if overall_mape < 5 else 'good' if overall_mape < 10 else 'moderate'} prediction accuracy.")
    
    if overall_r2 > 0.8:
        st.success("🎉 Model shows strong predictive power with high R² score!")
    elif overall_r2 > 0.6:
        st.info("📊 Model shows decent predictive capability.")
    else:
        st.warning("⚠️ Model predictions should be used with caution due to lower R² score.")
    
    st.write("**Disclaimer:** This prediction is based on historical data and technical indicators. Always conduct thorough research and consider multiple factors before making investment decisions.")