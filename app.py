import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from utils.preprocessing_simple import load_data, prepare_features
import datetime
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="📈 ML Stock Price Prediction Project", layout="wide")

# ==============================================================================
# 1. PROJECT STATEMENT
# ==============================================================================
st.title("📊 Machine Learning Stock Price Prediction Project")
st.markdown("---")

with st.expander("📋 **1. PROJECT STATEMENT**", expanded=True):
    st.markdown("""
    ### 🎯 **Project Objective**
    This project aims to predict stock prices using various machine learning algorithms and technical indicators. 
    The system analyzes historical stock data, applies feature engineering techniques, and compares multiple ML models 
    to provide accurate price predictions and trend analysis.
    
    ### 🔍 **Key Goals:**
    - Predict future stock prices using historical data and technical indicators
    - Compare performance of different ML algorithms (Regression & Classification)
    - Provide actionable insights for investment decision-making
    - Implement comprehensive model evaluation and validation
    
    ### 📈 **Business Impact:**
    - Enable data-driven investment decisions
    - Reduce investment risk through predictive analytics
    - Optimize portfolio management strategies
    """)

# ==============================================================================
# 2. DATASET DESCRIPTION
# ==============================================================================
with st.expander("📊 **2. DATASET DESCRIPTION**"):
    st.markdown("""
    ### 📋 **Data Source:** Alpha Vantage API
    - **Type:** Time Series Financial Data
    - **Frequency:** Daily stock prices
    - **Coverage:** Full historical data available
    
    ### 🏢 **Available Stocks:**
    - RELIANCE.BSE (Reliance Industries)
    - TCS.BSE (Tata Consultancy Services)
    - INFY.BSE (Infosys Limited)
    - WIPRO.BSE (Wipro Limited)
    
    ### 📈 **Core Features:**
    - **OHLCV Data:** Open, High, Low, Close, Volume
    - **Technical Indicators:** RSI, MACD, Bollinger Bands, Moving Averages
    - **Derived Features:** Momentum, Volatility, Price Ratios
    - **Time Features:** Day of week, Month, Day of month
    
    ### 🎯 **Target Variables:**
    - **Regression:** Future stock prices (continuous)
    - **Classification:** Price movement direction (up/down)
    """)

# Sidebar Configuration
st.sidebar.header("🔧 Model Configuration")
stock = st.sidebar.selectbox("Select Stock", ['RELIANCE.BSE', 'TCS.BSE', 'INFY.BSE', 'WIPRO.BSE'])
task_type = st.sidebar.selectbox("Task Type", ['Regression', 'Classification'])
n_days = st.sidebar.slider("Prediction Horizon (days)", min_value=1, max_value=30, value=5)
n_features = st.sidebar.slider("Number of Features", min_value=5, max_value=50, value=20)

# Model selection based on task type
st.sidebar.subheader("🤖 Model Selection")
if task_type == 'Regression':
    model_options = ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                    'ElasticNet', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVR', 'KNN']
else:
    model_options = ['Logistic Regression', 'Decision Tree', 'Random Forest', 
                    'XGBoost', 'SVM', 'KNN', 'Gaussian Naive Bayes']

model_type = st.sidebar.selectbox("Choose ML Model", model_options)

# Advanced options
st.sidebar.subheader("🧠 Advanced Options")
test_size = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
use_feature_selection = st.sidebar.checkbox("Use Feature Selection", value=True)
use_lagged_features = st.sidebar.checkbox("Add Lagged Features", value=True)
lag_periods = st.sidebar.multiselect("Lag Periods", [1, 2, 3, 5, 10], default=[1, 2, 3])

# Model-specific parameters
if model_type in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
    alpha = st.sidebar.slider("Regularization Strength (Alpha)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    if model_type == 'ElasticNet':
        l1_ratio = st.sidebar.slider("L1 Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if model_type in ['Random Forest', 'Decision Tree']:
    n_estimators = st.sidebar.slider("Number of Trees", min_value=50, max_value=500, value=100) if 'Forest' in model_type else None
    max_depth = st.sidebar.slider("Max Depth", min_value=3, max_value=20, value=10)

if model_type == 'XGBoost':
    n_estimators_xgb = st.sidebar.slider("N Estimators", min_value=50, max_value=500, value=100)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)

if model_type in ['KNN']:
    n_neighbors = st.sidebar.slider("Number of Neighbors", min_value=3, max_value=20, value=5)

# Helper Functions
def create_lagged_features(df, target_col='Close', lag_periods=[1, 2, 3]):
    """Create lagged features for time series prediction"""
    df_lagged = df.copy()
    
    for lag in lag_periods:
        df_lagged[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        df_lagged[f'{target_col}_diff_{lag}'] = df[target_col].diff(lag)
        df_lagged[f'{target_col}_pct_change_{lag}'] = df[target_col].pct_change(lag)
    
    return df_lagged

def create_classification_target(prices, n_days=1):
    """Create classification targets (price will go up or down)"""
    future_prices = prices.shift(-n_days)
    return (future_prices > prices).astype(int)

def get_model(model_type, task_type, **kwargs):
    """Get the selected ML model"""
    if task_type == 'Regression':
        if model_type == 'Linear Regression':
            return LinearRegression()
        elif model_type == 'Ridge Regression':
            return Ridge(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'Lasso Regression':
            return Lasso(alpha=kwargs.get('alpha', 1.0))
        elif model_type == 'ElasticNet':
            return ElasticNet(alpha=kwargs.get('alpha', 1.0), l1_ratio=kwargs.get('l1_ratio', 0.5))
        elif model_type == 'Decision Tree':
            return DecisionTreeRegressor(max_depth=kwargs.get('max_depth', 10), random_state=42)
        elif model_type == 'Random Forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'XGBoost':
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        elif model_type == 'SVR':
            return SVR(kernel='rbf', C=kwargs.get('C', 1.0))
        elif model_type == 'KNN':
            return KNeighborsRegressor(n_neighbors=kwargs.get('n_neighbors', 5))
    
    else:  # Classification
        if model_type == 'Logistic Regression':
            return LogisticRegression(random_state=42)
        elif model_type == 'Decision Tree':
            return DecisionTreeClassifier(max_depth=kwargs.get('max_depth', 10), random_state=42)
        elif model_type == 'Random Forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'XGBoost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        elif model_type == 'SVM':
            return SVC(kernel='rbf', C=kwargs.get('C', 1.0), random_state=42)
        elif model_type == 'KNN':
            return KNeighborsClassifier(n_neighbors=kwargs.get('n_neighbors', 5))
        elif model_type == 'Gaussian Naive Bayes':
            return GaussianNB()

def calculate_regression_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1e-8))) * 100
    
    return {
        'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE': mape
    }

def calculate_classification_metrics(y_true, y_pred):
    """Calculate comprehensive classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    return {
        'Accuracy': accuracy,
        'Classification Report': classification_report(y_true, y_pred),
        'Confusion Matrix': confusion_matrix(y_true, y_pred)
    }

# Main Application
if st.button("🚀 Start ML Analysis", type="primary"):
    
    # ==============================================================================
    # 3. DATA INGESTION
    # ==============================================================================
    st.header("3️⃣ Data Ingestion")
    with st.spinner("📊 Loading stock data from Alpha Vantage API..."):
        try:
            df = load_data(stock)
            st.success(f"✅ Successfully loaded {len(df)} records for {stock}")
            
            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Date Range", f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            with col3:
                st.metric("Features", len(df.columns))
                
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            st.stop()
    
    # ==============================================================================
    # 4. IMPORT REQUIRED LIBRARIES (Already done at the top)
    # ==============================================================================
    
    # ==============================================================================
    # 5. BASIC DATA INSPECTION
    # ==============================================================================
    st.header("5️⃣ Basic Data Inspection")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Dataset Info")
        buffer = []
        buffer.append(f"Shape: {df.shape}")
        buffer.append(f"Columns: {list(df.columns)}")
        buffer.append(f"Data Types: {df.dtypes.value_counts().to_dict()}")
        buffer.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        st.text("\n".join(buffer))
        
        st.subheader("📊 Missing Values")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.dataframe(missing_data[missing_data > 0])
        else:
            st.success("✅ No missing values found!")
    
    with col2:
        st.subheader("📈 Raw Data Sample")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].head(10))
        
        st.subheader("📊 Statistical Summary")
        st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].describe())
    
    # ==============================================================================
    # 6. EXPLORATORY DATA ANALYSIS (EDA)
    # ==============================================================================
    st.header("6️⃣ Exploratory Data Analysis (EDA)")
    
    # Price Movement Visualization
    st.subheader("📈 Interactive Price Chart")
    last_year_data = df.iloc[-252:] if len(df) >= 252 else df
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{stock} - Price Movement', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=last_year_data.index,
            open=last_year_data['Open'],
            high=last_year_data['High'],
            low=last_year_data['Low'],
            close=last_year_data['Close'],
            name="Price"
        ), row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=last_year_data.index,
            y=last_year_data['Volume'],
            name="Volume",
            opacity=0.6
        ), row=2, col=1
    )
    
    fig.update_layout(height=600, title=f"{stock} - Interactive Price & Volume Chart")
    st.plotly_chart(fig, use_container_width=True)
    
    # Price Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Price Distribution")
        fig = px.histogram(df, x='Close', nbins=50, title="Close Price Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Returns Distribution")
        returns = df['Close'].pct_change().dropna()
        fig = px.histogram(returns, nbins=50, title="Daily Returns Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    st.subheader("🔗 Correlation Analysis")
    corr_matrix = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Feature Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================================================
    # 7. DATA TRANSFORMATION
    # ==============================================================================
    st.header("7️⃣ Data Transformation")
    
    with st.spinner("🔧 Engineering features and transforming data..."):
        # Prepare features
        feature_df = prepare_features(df)
        
        # Add lagged features if selected
        if use_lagged_features and lag_periods:
            feature_df = create_lagged_features(feature_df, 'Close', lag_periods)
            st.write(f"✅ Added lagged features for periods: {lag_periods}")
        
        # Remove NaN values
        feature_df = feature_df.dropna()
        
        # Feature selection
        if use_feature_selection:
            correlations = feature_df.corr()['Close'].abs().sort_values(ascending=False)
            top_features = correlations.head(n_features).index.tolist()
        else:
            top_features = [col for col in feature_df.columns if col != 'Close'][:n_features]
            top_features.append('Close')
        
        selected_features = feature_df[top_features].copy()
        
        st.success(f"✅ Feature engineering completed. {len(selected_features.columns)} features selected.")
        
        # Display feature importance
        if use_feature_selection:
            st.subheader("🎯 Feature Importance (Correlation with Target)")
            feature_importance = correlations[top_features[:-1]].sort_values(ascending=False)
            fig = px.bar(
                x=feature_importance.values,
                y=feature_importance.index,
                orientation='h',
                title="Feature Correlation with Close Price"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================================================
    # 8. SPLIT X, y
    # ==============================================================================
    st.header("8️⃣ Split Features and Target")
    
    X = selected_features.drop('Close', axis=1)
    
    if task_type == 'Regression':
        y = selected_features['Close']
        st.write(f"📊 **Regression Task**: Predicting continuous stock prices")
    else:
        y = create_classification_target(selected_features['Close'], n_days=1)
        # Align X and y (remove last row due to future target creation)
        X = X.iloc[:-1]
        y = y.iloc[:-1]
        st.write(f"📊 **Classification Task**: Predicting price direction (Up/Down)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Feature Matrix Shape", f"{X.shape[0]} × {X.shape[1]}")
    with col2:
        if task_type == 'Classification':
            st.metric("Class Distribution", f"Up: {y.sum()}, Down: {len(y) - y.sum()}")
        else:
            st.metric("Target Range", f"{y.min():.2f} - {y.max():.2f}")
    
    # ==============================================================================
    # 9. TRAIN-TEST SPLIT
    # ==============================================================================
    st.header("9️⃣ Train-Test Split")
    
    # For time series, use temporal split
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Samples", len(X_train))
    with col2:
        st.metric("Testing Samples", len(X_test))
    with col3:
        st.metric("Test Ratio", f"{test_size:.1%}")
    
    st.success("✅ Data split completed with temporal ordering preserved")
    
    # ==============================================================================
    # 10. MODEL SELECTION (Already implemented in sidebar)
    # ==============================================================================
    
    # ==============================================================================
    # 11. MODEL BUILDING
    # ==============================================================================
    st.header("🏗️ Model Building")
    
    with st.spinner(f"🤖 Training {model_type} model..."):
        # Get model parameters
        model_params = {}
        
        if model_type in ['Ridge Regression', 'Lasso Regression', 'ElasticNet']:
            model_params['alpha'] = alpha
            if model_type == 'ElasticNet':
                model_params['l1_ratio'] = l1_ratio
        
        if model_type in ['Random Forest', 'Decision Tree']:
            if 'Forest' in model_type:
                model_params['n_estimators'] = n_estimators
            model_params['max_depth'] = max_depth
        
        if model_type == 'XGBoost':
            model_params['n_estimators'] = n_estimators_xgb
            model_params['learning_rate'] = learning_rate
        
        if model_type in ['KNN']:
            model_params['n_neighbors'] = n_neighbors
        
        # Create and train model
        model = get_model(model_type, task_type, **model_params)
        model.fit(X_train_scaled, y_train)
        
        st.success(f"✅ {model_type} model trained successfully!")
        
        # Display model parameters
        st.subheader("⚙️ Model Configuration")
        config_info = {
            'Model Type': model_type,
            'Task': task_type,
            'Features': X_train.shape[1],
            'Training Samples': len(X_train),
            'Test Samples': len(X_test)
        }
        config_info.update(model_params)
        
        config_df = pd.DataFrame(list(config_info.items()), columns=['Parameter', 'Value'])
        st.dataframe(config_df, use_container_width=True)
    
    # ==============================================================================
    # 12. MODEL PREDICTION
    # ==============================================================================
    st.header("🔮 Model Prediction")
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Predictions", len(y_pred_train))
    with col2:
        st.metric("Testing Predictions", len(y_pred_test))
    
    # ==============================================================================
    # 13. MODEL EVALUATION
    # ==============================================================================
    st.header("📊 Model Evaluation")
    
    if task_type == 'Regression':
        # Regression metrics
        train_metrics = calculate_regression_metrics(y_train, y_pred_train)
        test_metrics = calculate_regression_metrics(y_test, y_pred_test)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏋️ Training Performance")
            train_df = pd.DataFrame(list(train_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(train_df)
        
        with col2:
            st.subheader("🎯 Testing Performance")
            test_df = pd.DataFrame(list(test_metrics.items()), columns=['Metric', 'Value'])
            st.dataframe(test_df)
        
        # Prediction vs Actual Plot
        st.subheader("📈 Prediction vs Actual")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(y_test))),
            y=y_test,
            mode='lines+markers',
            name='Actual',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(y_pred_test))),
            y=y_pred_test,
            mode='lines+markers',
            name='Predicted',
            line=dict(color='red', dash='dot')
        ))
        fig.update_layout(
            title=f'{stock} - {model_type} Predictions vs Actual',
            xaxis_title='Time',
            yaxis_title='Price (₹)',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual Analysis
        residuals = y_test - y_pred_test
        fig = px.scatter(x=y_pred_test, y=residuals, title="Residual Plot")
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # Classification metrics
        train_metrics = calculate_classification_metrics(y_train, y_pred_train)
        test_metrics = calculate_classification_metrics(y_test, y_pred_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🏋️ Training Accuracy")
            st.metric("Accuracy", f"{train_metrics['Accuracy']:.4f}")
        
        with col2:
            st.subheader("🎯 Testing Accuracy")
            st.metric("Accuracy", f"{test_metrics['Accuracy']:.4f}")
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = test_metrics['Confusion Matrix']
        fig = px.imshow(cm, text_auto=True, aspect="auto", 
                       title="Confusion Matrix", 
                       labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📋 Classification Report")
        st.text(test_metrics['Classification Report'])
    
    # Feature Importance (for applicable models)
    if hasattr(model, 'feature_importances_'):
        st.subheader("🎯 Feature Importance")
        importance = model.feature_importances_
        feature_names = X.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(15),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Feature Importance'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==============================================================================
    # 14. RECOMMENDATIONS/INSIGHTS
    # ==============================================================================
    st.header("💡 Recommendations & Insights")
    
    # Calculate key insights
    if task_type == 'Regression':
        model_performance = test_metrics['R²']
        accuracy_measure = f"R² Score: {model_performance:.4f}"
        performance_rating = "Excellent" if model_performance > 0.8 else "Good" if model_performance > 0.6 else "Moderate"
    else:
        model_performance = test_metrics['Accuracy']
        accuracy_measure = f"Accuracy: {model_performance:.4f}"
        performance_rating = "Excellent" if model_performance > 0.8 else "Good" if model_performance > 0.6 else "Moderate"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance Summary")
        st.write(f"**Selected Model:** {model_type}")
        st.write(f"**Task Type:** {task_type}")
        st.write(f"**Performance:** {accuracy_measure}")
        st.write(f"**Rating:** {performance_rating}")
        st.write(f"**Features Used:** {len(X.columns)}")
        st.write(f"**Training Period:** {X_train.shape[0]} days")
        st.write(f"**Testing Period:** {X_test.shape[0]} days")
    
    with col2:
        st.subheader("🎯 Key Insights")
        
        insights = []
        
        if task_type == 'Regression':
            if test_metrics['MAPE'] < 5:
                insights.append("🟢 **Low MAPE**: Model shows excellent prediction accuracy")
            elif test_metrics['MAPE'] < 10:
                insights.append("🟡 **Moderate MAPE**: Model shows decent prediction accuracy")
            else:
                insights.append("🔴 **High MAPE**: Model predictions should be used with caution")
        
        if hasattr(model, 'feature_importances_'):
            top_feature = feature_names[np.argmax(importance)]
            insights.append(f"📈 **Most Important Feature**: {top_feature}")
        
        if task_type == 'Classification':
            if test_metrics['Accuracy'] > 0.7:
                insights.append("🎯 **Good Classification**: Model can effectively predict price direction")
            else:
                insights.append("⚠️ **Moderate Classification**: Consider feature engineering or model tuning")
        
        for insight in insights:
            st.write(insight)
    
    # Actionable Recommendations
    st.subheader("🚀 Actionable Recommendations")
    
    recommendations = []
    
    if task_type == 'Regression' and test_metrics['R²'] < 0.6:
        recommendations.append("📊 **Model Improvement**: Consider ensemble methods or feature engineering")
    
    if task_type == 'Classification' and test_metrics['Accuracy'] < 0.7:
        recommendations.append("🔄 **Strategy Adjustment**: Try different algorithms or hyperparameter tuning")
    
    recommendations.extend([
        "💼 **Investment Strategy**: Use predictions as one factor in a diversified approach",
        "📈 **Risk Management**: Set stop-loss orders based on model confidence intervals",
        "⏰ **Time Horizon**: Short-term predictions are generally more reliable than long-term",
        "🔄 **Model Updates**: Retrain the model periodically with new market data",
        "📊 **Portfolio Diversification**: Don't rely solely on single stock predictions",
        "🎯 **Backtesting**: Validate strategy performance on historical out-of-sample data",
        "⚠️ **Market Conditions**: Consider external factors like market volatility and news",
        "💡 **Feature Enhancement**: Incorporate additional technical indicators or sentiment data",
        "🔍 **Model Validation**: Use cross-validation for more robust performance estimates",
        "📱 **Real-time Monitoring**: Implement alerts for significant prediction deviations"
    ])
    
    # Display recommendations
    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")
    
    # Additional Trading Guidelines
    st.subheader("⚡ Trading Guidelines")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🟢 When to Consider Buying:**
        - Model shows strong upward prediction confidence
        - Multiple technical indicators align
        - Market sentiment is positive
        - Risk-reward ratio is favorable (>1:2)
        """)
    
    with col2:
        st.markdown("""
        **🔴 When to Consider Selling:**
        - Model predicts significant downward movement
        - Stop-loss levels are triggered
        - Market shows high volatility
        - Fundamental factors change negatively
        """)
    
    # Model Limitations
    st.subheader("⚠️ Model Limitations & Disclaimers")
    
    st.warning("""
    **Important Disclaimers:**
    
    🚨 **Not Financial Advice**: This model is for educational purposes only and should not be considered as financial advice.
    
    📊 **Past Performance**: Historical performance does not guarantee future results.
    
    🌊 **Market Volatility**: Stock markets are inherently unpredictable and influenced by numerous external factors.
    
    🔄 **Model Drift**: Model performance may degrade over time as market conditions change.
    
    💰 **Risk Warning**: All investments carry risk, and you may lose money. Never invest more than you can afford to lose.
    
    📈 **Diversification**: Always maintain a diversified portfolio and consult with financial professionals.
    """)
    
    # Performance Summary Dashboard
    st.subheader("📊 Performance Summary Dashboard")
    
    # Create performance summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if task_type == 'Regression':
            st.metric(
                "Model Accuracy", 
                f"{test_metrics['R²']:.3f}",
                delta=f"{test_metrics['R²'] - 0.5:.3f}" if test_metrics['R²'] > 0.5 else None
            )
        else:
            st.metric(
                "Classification Accuracy", 
                f"{test_metrics['Accuracy']:.3f}",
                delta=f"{test_metrics['Accuracy'] - 0.5:.3f}" if test_metrics['Accuracy'] > 0.5 else None
            )
    
    with col2:
        if task_type == 'Regression':
            st.metric("MAPE", f"{test_metrics['MAPE']:.2f}%")
        else:
            precision = len(y_test[y_test == y_pred_test]) / len(y_test)
            st.metric("Precision", f"{precision:.3f}")
    
    with col3:
        volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
        st.metric("Annual Volatility", f"{volatility:.1f}%")
    
    with col4:
        sharpe_ratio = df['Close'].pct_change().mean() / df['Close'].pct_change().std() * np.sqrt(252)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    # Future Enhancements Section
    st.subheader("🔮 Future Enhancements")
    
    future_enhancements = [
        "🤖 **Deep Learning**: Implement LSTM/GRU models for sequential pattern recognition",
        "📰 **Sentiment Analysis**: Incorporate news sentiment and social media data",
        "🌐 **Multi-Asset**: Extend to portfolio optimization across multiple stocks",
        "📱 **Real-time API**: Develop live trading signal generation",
        "🎯 **Advanced Features**: Add options pricing, earnings data, and macro-economic indicators",
        "🔄 **Auto-retraining**: Implement automated model retraining pipeline",
        "📊 **Advanced Visualization**: Create interactive trading dashboards",
        "⚡ **High-frequency**: Develop intraday and minute-level prediction models"
    ]
    
    for enhancement in future_enhancements:
        st.write(f"• {enhancement}")
    
    # Success message
    st.success("""
    🎉 **Analysis Complete!** 
    
    Your machine learning stock prediction model has been successfully built and evaluated. 
    Remember to use these insights responsibly and as part of a comprehensive investment strategy.
    """)
    
    # Download results option
    if st.button("📥 Download Analysis Report"):
        # Create a comprehensive report
        report_data = {
            'Model Configuration': {
                'Stock': stock,
                'Model Type': model_type,
                'Task': task_type,
                'Features': len(X.columns),
                'Training Samples': len(X_train),
                'Test Samples': len(X_test)
            },
            'Performance Metrics': test_metrics,
            'Recommendations': recommendations
        }
        
        # Convert to JSON for download (in a real app, you'd create a PDF or CSV)
        st.json(report_data)
        st.info("💡 In a production environment, this would generate a downloadable PDF report.")