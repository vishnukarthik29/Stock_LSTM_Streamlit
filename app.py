import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from utils.preprocessing_simple import load_data, prepare_features
import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="📈 Linear Regression Stock Price Prediction", layout="wide")
st.title("📊 Stock Price Prediction using Linear Regression")

# Sidebar
st.sidebar.header("🔧 Model Configuration")
stock = st.sidebar.selectbox("Select Stock", ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE', 'WIPRO.BSE'])
n_days = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=30, value=5)
n_features = st.sidebar.slider("Number of Features", min_value=5, max_value=50, value=20)

# Model selection
st.sidebar.subheader("🤖 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Regression Model", 
    ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet', 'Random Forest']
)

# Advanced options
st.sidebar.subheader("🧠 Advanced Options")
if model_type in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
    alpha = st.sidebar.slider("Regularization Strength (Alpha)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    if model_type == 'ElasticNet':
        l1_ratio = st.sidebar.slider("L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if model_type == 'Random Forest':
    n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100)
    max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=20, value=10)

# Feature engineering options
use_feature_selection = st.sidebar.checkbox("Use Feature Selection", value=True)
use_lagged_features = st.sidebar.checkbox("Add Lagged Features", value=True)
lag_periods = st.sidebar.multiselect("Lag Periods", [1, 2, 3, 5, 10], default=[1, 2, 3])

def create_lagged_features(df, target_col='Close', lag_periods=[1, 2, 3]):
    """Create lagged features for time series prediction"""
    df_lagged = df.copy()
    
    for lag in lag_periods:
        df_lagged[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        df_lagged[f'{target_col}_diff_{lag}'] = df[target_col].diff(lag)
        df_lagged[f'{target_col}_pct_change_{lag}'] = df[target_col].pct_change(lag)
    
    return df_lagged

def create_target_sequences(target, n_days):
    """Create multi-step prediction targets"""
    targets = []
    for i in range(len(target) - n_days + 1):
        targets.append(target[i:i+n_days])
    return np.array(targets)

def get_model(model_type, **kwargs):
    """Get the selected regression model"""
    if model_type == 'Linear Regression':
        return LinearRegression()
    elif model_type == 'Ridge Regression':
        return Ridge(alpha=kwargs.get('alpha', 1.0))
    elif model_type == 'Lasso Regression':
        return Lasso(alpha=kwargs.get('alpha', 1.0))
    elif model_type == 'ElasticNet':
        return ElasticNet(alpha=kwargs.get('alpha', 1.0), l1_ratio=kwargs.get('l1_ratio', 0.5))
    elif model_type == 'Random Forest':
        return RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            random_state=42
        )

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Handle division by zero for MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100
    
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
            st.subheader("📊 Interactive Price Visualization")
            
            # Create interactive plotly chart
            last_year_data = df.iloc[-252:] if len(df) >= 252 else df
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{stock} - Price Movement', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=last_year_data.index,
                    open=last_year_data['Open'],
                    high=last_year_data['High'],
                    low=last_year_data['Low'],
                    close=last_year_data['Close'],
                    name="Price",
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
            
            # Add moving averages
            if 'MA_20' in feature_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=last_year_data.index,
                        y=feature_df['MA_20'].iloc[-252:] if len(df) >= 252 else feature_df['MA_20'],
                        mode='lines',
                        name='MA-20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'MA_50' in feature_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=last_year_data.index,
                        y=feature_df['MA_50'].iloc[-252:] if len(df) >= 252 else feature_df['MA_50'],
                        mode='lines',
                        name='MA-50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # Add volume chart
            colors = ['green' if close >= open else 'red' for close, open in 
                     zip(last_year_data['Close'], last_year_data['Open'])]
            
            fig.add_trace(
                go.Bar(
                    x=last_year_data.index,
                    y=last_year_data['Volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.6
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"{stock} - Interactive Price Chart",
                yaxis_title="Price (₹)",
                yaxis2_title="Volume",
                xaxis_title="Date",
                height=600,
                showlegend=True,
                hovermode='x unified',
                xaxis_rangeslider_visible=False
            )
            
            # Add range selector
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=False),
                    type="date"
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Feature engineering
    with st.spinner("🔧 Engineering features..."):
        # Add lagged features if selected
        if use_lagged_features and lag_periods:
            feature_df = create_lagged_features(feature_df, 'Close', lag_periods)
            st.write(f"✅ Added lagged features for periods: {lag_periods}")
        
        # Ensure Close column exists
        if 'Close' not in feature_df.columns:
            st.error("❌ 'Close' column not found in feature data. Please check data preprocessing.")
            st.stop()
        
        # Remove rows with NaN values (created by lagging)
        feature_df = feature_df.dropna()
        
        # Feature selection
        if use_feature_selection:
            # Select top features based on correlation with target
            correlations = feature_df.corr()['Close'].abs().sort_values(ascending=False)
            top_features = correlations.head(n_features).index.tolist()
        else:
            # Use all available features except Close
            top_features = [col for col in feature_df.columns if col != 'Close'][:n_features]
            top_features.append('Close')
        
        # Ensure Close is included
        if 'Close' not in top_features:
            top_features.append('Close')
        
        selected_features = feature_df[top_features].copy()
        
        st.subheader("🎯 Selected Features")
        st.write(f"Top {len(top_features)} features selected:")
        if use_feature_selection:
            feature_importance = correlations[top_features].sort_values(ascending=False)
            
            # Create interactive bar chart for feature importance
            fig = px.bar(
                x=feature_importance[1:].values,  # Exclude Close itself
                y=feature_importance[1:].index,
                orientation='h',
                title="Feature Correlation with Close Price",
                labels={'x': 'Correlation Coefficient', 'y': 'Features'},
                color=feature_importance[1:].values,
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Data preprocessing
    with st.spinner("⚙️ Preprocessing data..."):
        # Separate features and target
        X = selected_features.drop('Close', axis=1)
        y = selected_features['Close']
        
        # Create multi-step targets
        y_sequences = create_target_sequences(y.values, n_days)
        
        # Align X with y_sequences (remove last n_days-1 rows from X)
        X_aligned = X.iloc[:len(y_sequences)]
        
        # Use RobustScaler for better handling of outliers
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
        
        # Scale features
        X_scaled = feature_scaler.fit_transform(X_aligned)
        y_scaled = target_scaler.fit_transform(y_sequences)
        
        st.write(f"✅ Feature matrix shape: {X_scaled.shape}")
        st.write(f"✅ Target matrix shape: {y_scaled.shape}")
    
    # Train-test split (time series aware)
    split_ratio = 0.8
    split_index = int(len(X_scaled) * split_ratio)
    
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
    
    # Model training
    st.subheader(f"🤖 Training {model_type} Model")
    
    with st.spinner("🏗️ Training models..."):
        models = {}
        predictions = {}
        
        # Train separate models for each prediction day
        for day in range(n_days):
            # Get model parameters
            model_params = {}
            if model_type in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
                model_params['alpha'] = alpha
                if model_type == 'ElasticNet':
                    model_params['l1_ratio'] = l1_ratio
            elif model_type == 'Random Forest':
                model_params['n_estimators'] = n_estimators
                model_params['max_depth'] = max_depth
            
            # Create and train model
            model = get_model(model_type, **model_params)
            model.fit(X_train, y_train[:, day])
            
            # Make predictions
            pred = model.predict(X_test)
            
            models[f'Day_{day+1}'] = model
            predictions[f'Day_{day+1}'] = pred
        
        st.write(f"✅ Trained {n_days} {model_type} models")
    
    # Feature importance (for applicable models)
    if model_type == 'Random Forest':
        st.subheader("🎯 Feature Importance")
        feature_names = X.columns
        
        # Average feature importance across all day models
        avg_importance = np.mean([models[f'Day_{day+1}'].feature_importances_ for day in range(n_days)], axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': avg_importance
        }).sort_values('Importance', ascending=False)
        
        # Create interactive feature importance chart
        fig = px.bar(
            importance_df[:15],
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Feature Importance (Random Forest)',
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Predictions and evaluation
    st.subheader("🔮 Model Predictions")
    
    with st.spinner("📊 Generating predictions..."):
        # Combine predictions
        y_pred_scaled = np.column_stack([predictions[f'Day_{day+1}'] for day in range(n_days)])
        
        # Transform back to original scale
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        y_actual = target_scaler.inverse_transform(y_test)
        
        # Calculate metrics for each prediction day
        metrics_by_day = {}
        for day in range(n_days):
            day_metrics = calculate_metrics(y_actual[:, day], y_pred[:, day])
            metrics_by_day[f'Day {day+1}'] = day_metrics
    
    # Visualize predictions
    st.subheader("📈 Interactive Prediction vs Actual")
    
    # Create interactive prediction charts
    for i in range(min(n_days, 3)):
        st.write(f"**Day {i+1} Prediction Results**")
        
        # Create subplot with actual vs predicted
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_actual[:, i]))),
                y=y_actual[:, i],
                mode='lines+markers',
                name=f'Actual Day {i+1}',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                hovertemplate='Sample: %{x}<br>Actual Price: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        # Add predicted values
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_pred[:, i]))),
                y=y_pred[:, i],
                mode='lines+markers',
                name=f'Predicted Day {i+1}',
                line=dict(color='red', width=2, dash='dot'),
                marker=dict(size=4),
                hovertemplate='Sample: %{x}<br>Predicted Price: ₹%{y:.2f}<extra></extra>'
            )
        )
        
        # Calculate R² for this day
        r2_day = r2_score(y_actual[:, i], y_pred[:, i])
        
        fig.update_layout(
            title=f'{stock} - Day {i+1} Prediction ({model_type}) | R² = {r2_day:.4f}',
            xaxis_title='Test Samples',
            yaxis_title='Price (₹)',
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Metrics display
    st.subheader("📏 Model Performance Metrics")
    
    metrics_df = pd.DataFrame(metrics_by_day).T
    st.dataframe(metrics_df.round(4))
    
    # Overall performance
    overall_mape = np.mean([metrics_by_day[day]['MAPE'] for day in metrics_by_day.keys()])
    overall_r2 = np.mean([metrics_by_day[day]['R²'] for day in metrics_by_day.keys()])
    overall_rmse = np.mean([metrics_by_day[day]['RMSE'] for day in metrics_by_day.keys()])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall MAPE", f"{overall_mape:.2f}%")
    with col2:
        st.metric("Overall R² Score", f"{overall_r2:.4f}")
    with col3:
        st.metric("Overall RMSE", f"{overall_rmse:.2f}")
    with col4:
        accuracy = max(0, 100 - overall_mape)
        st.metric("Prediction Accuracy", f"{accuracy:.1f}%")
    
    # Future prediction with interactive chart
    st.subheader("🔮 Interactive Future Price Prediction")
    
    # Use the last sample to predict future prices
    last_sample = X_scaled[-1:] if len(X_scaled) > 0 else X_scaled[-1].reshape(1, -1)
    future_pred_scaled = np.array([models[f'Day_{day+1}'].predict(last_sample)[0] for day in range(n_days)])
    future_pred = target_scaler.inverse_transform(future_pred_scaled.reshape(1, -1))[0]
    
    current_price = y.iloc[-1]
    
    # Create interactive future prediction chart
    dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_days, freq='D')
    
    fig = go.Figure()
    
    # Add historical prices (last 30 days)
    historical_data = df.tail(30)
    fig.add_trace(
        go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines+markers',
            name='Historical Prices',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        )
    )
    
    # Add current price point
    fig.add_trace(
        go.Scatter(
            x=[df.index[-1]],
            y=[current_price],
            mode='markers',
            name='Current Price',
            marker=dict(size=10, color='green', symbol='diamond')
        )
    )
    
    # Add predicted prices
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=future_pred,
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='red', width=2, dash='dot'),
            marker=dict(size=6),
            hovertemplate='Date: %{x}<br>Predicted Price: ₹%{y:.2f}<extra></extra>'
        )
    )
    
    # Add connecting line
    fig.add_trace(
        go.Scatter(
            x=[df.index[-1], dates[0]],
            y=[current_price, future_pred[0]],
            mode='lines',
            name='Transition',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title=f'{stock} - Future Price Prediction ({model_type})',
        xaxis_title='Date',
        yaxis_title='Price (₹)',
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.write("🎯 **Next Few Days Prediction:**")
    
    # Create a summary table
    prediction_data = []
    for i, (date, price) in enumerate(zip(dates, future_pred)):
        change = ((price - current_price) / current_price) * 100
        prediction_data.append({
            'Day': f'Day {i+1}',
            'Date': date.strftime('%Y-%m-%d'),
            'Predicted Price': f'₹{price:.2f}',
            'Change': f'{change:+.2f}%',
            'Trend': '🟢 Up' if change > 0 else '🔴 Down' if change < 0 else '🟡 Flat'
        })
    
    prediction_df = pd.DataFrame(prediction_data)
    st.dataframe(prediction_df, use_container_width=True)
    
    # Model coefficients (for linear models)
    if model_type in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']:
        st.subheader("📊 Model Coefficients")
        
        # Show coefficients for Day 1 model as example
        coefficients = models['Day_1'].coef_
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': coefficients
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        st.dataframe(coef_df.round(6))
    
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
    
    st.write(f"**Model Used:** {model_type}")
    st.write(f"**Trend Analysis:** The model predicts a {trend} trend over the next {n_days} days.")
    st.write(f"**Confidence Level:** {confidence_color} {confidence_level} ({confidence:.1f}%)")
    st.write(f"**Model Performance:** MAPE of {overall_mape:.2f}% indicates {'excellent' if overall_mape < 5 else 'good' if overall_mape < 10 else 'moderate'} prediction accuracy.")
    
    if overall_r2 > 0.8:
        st.success("🎉 Model shows strong predictive power with high R² score!")
    elif overall_r2 > 0.6:
        st.info("📊 Model shows decent predictive capability.")
    else:
        st.warning("⚠️ Model predictions should be used with caution due to lower R² score.")
    
    # Model-specific insights
    if model_type == 'Random Forest':
        st.write("**Random Forest Benefits:** Handles non-linear relationships and feature interactions well.")
    elif model_type in ['Ridge Regression', 'Lasso Regression']:
        st.write(f"**Regularization Benefits:** {model_type} helps prevent overfitting with alpha={alpha}.")
    elif model_type == 'Linear Regression':
        st.write("**Linear Model:** Assumes linear relationships between features and target.")
    
    st.write("**Disclaimer:** This prediction is based on historical data and technical indicators. Always conduct thorough research and consider multiple factors before making investment decisions.")