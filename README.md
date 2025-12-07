# 📈 Stock Price Prediction using Linear Regression

A comprehensive Streamlit web application for predicting Indian stock prices using various regression models and technical indicators.

## 🌟 Features

### Core Functionality

- **Multiple Regression Models**: Linear Regression, Ridge, Lasso, ElasticNet, and Random Forest
- **Multi-Day Predictions**: Forecast stock prices up to 30 days ahead
- **Technical Indicators**: Integrated moving averages, RSI, MACD, Bollinger Bands, and more
- **Feature Engineering**: Automated lagged features and time-series preprocessing
- **Interactive Visualizations**: Candlestick charts, volume analysis, and prediction plots using Plotly

### Supported Stocks

- RELIANCE.BSE (Reliance Industries)
- TCS.BSE (Tata Consultancy Services)
- INFY.BSE (Infosys)
- WIPRO.BSE (Wipro)

### Model Options

1. **Linear Regression**: Fast, interpretable baseline model
2. **Ridge Regression**: L2 regularization to prevent overfitting
3. **Lasso Regression**: L1 regularization for feature selection
4. **ElasticNet**: Combined L1 and L2 regularization
5. **Random Forest**: Non-linear ensemble model with feature importance

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone <repository-url>
cd stock-price-prediction
```

2. **Create a virtual environment** (recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required packages**

```bash
pip install -r requirements.txt
```

### Required Dependencies

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
scikit-learn>=1.3.0
yfinance>=0.2.0
ta>=0.11.0
```

## 📁 Project Structure

```
stock-price-prediction/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── utils/
│   ├── __init__.py
│   └── preprocessing_simple.py     # Data loading and feature engineering
│
├── data/                           # Cached data directory (auto-created)
│
└── models/                         # Saved models (optional)
```

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app**

```bash
streamlit run app.py
```

2. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

### Using the Interface

#### Sidebar Configuration

1. **Stock Selection**

   - Choose from available BSE-listed stocks

2. **Prediction Horizon**

   - Set the number of days ahead to predict (1-30 days)

3. **Model Selection**

   - Choose your preferred regression model
   - Configure model-specific parameters:
     - **Ridge/Lasso/ElasticNet**: Adjust regularization strength (alpha)
     - **ElasticNet**: Set L1 ratio for regularization balance
     - **Random Forest**: Configure number of trees and max depth

4. **Advanced Options**

   - **Feature Selection**: Automatically select top N most correlated features
   - **Lagged Features**: Add historical lag periods (1, 2, 3, 5, 10 days)
   - **Number of Features**: Control feature set size

5. **Train and Predict**
   - Click the "🚀 Train and Predict" button to start the analysis

### Understanding the Output

#### 1. Data Overview

- Raw data statistics and recent prices
- Interactive candlestick chart with moving averages
- Volume analysis with color-coded bars

#### 2. Feature Importance

- Correlation heatmap showing feature relationships
- Top features ranked by correlation with closing price
- Random Forest feature importance (when applicable)

#### 3. Model Performance

- **Metrics per prediction day**:

  - MSE (Mean Squared Error)
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score (Coefficient of Determination)
  - MAPE (Mean Absolute Percentage Error)

- **Overall Performance**:
  - Average metrics across all prediction days
  - Prediction accuracy percentage
  - Confidence level indicators

#### 4. Predictions Visualization

- Actual vs Predicted price comparisons
- Day-by-day prediction accuracy
- Interactive charts with hover information

#### 5. Future Price Forecast

- Next N days price predictions
- Trend analysis (upward/downward)
- Expected price changes in percentage
- Date-wise prediction table

#### 6. AI Insights

- Trend direction and strength
- Confidence level assessment
- Model-specific recommendations
- Performance interpretation

## 📈 Performance Tips

### For Better Predictions

1. **Use more historical data**: 5+ years recommended
2. **Feature selection**: Enable for cleaner models
3. **Add lagged features**: Captures time-series patterns
4. **Try different models**: Compare performance across models
5. **Adjust regularization**: Fine-tune alpha parameters

### For Faster Training

1. **Reduce feature count**: Use fewer features
2. **Shorter prediction horizon**: Predict fewer days
3. **Use Linear/Ridge models**: Faster than Random Forest
4. **Disable feature selection**: Skip SelectKBest step

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No data found for stock"

- **Solution**: Check stock symbol format (e.g., 'RELIANCE.BSE' not 'RELIANCE')
- Verify internet connection for Yahoo Finance API

**Issue**: "NaN values in features"

- **Solution**: Increase historical data period
- Some indicators need minimum data points (e.g., MA_200 needs 200+ days)

**Issue**: "Memory error with Random Forest"

- **Solution**: Reduce number of trees or max depth
- Use fewer features
- Reduce training data size

**Issue**: "Poor prediction accuracy"

- **Solution**: Try different models
- Enable lagged features
- Increase number of features
- Check if stock is highly volatile

## 📝 Interpretation Guide

### R² Score (Coefficient of Determination)

- **> 0.8**: Excellent fit
- **0.6 - 0.8**: Good fit
- **0.4 - 0.6**: Moderate fit
- **< 0.4**: Poor fit

### MAPE (Mean Absolute Percentage Error)

- **< 5%**: Excellent accuracy
- **5% - 10%**: Good accuracy
- **10% - 20%**: Moderate accuracy
- **> 20%**: Poor accuracy

### Confidence Levels

- **High (🟢)**: MAPE < 5%, reliable predictions
- **Medium (🟡)**: MAPE 5-10%, decent predictions
- **Low (🔴)**: MAPE > 10%, use with caution

## ⚠️ Disclaimer

**This application is for educational and research purposes only.**

- Stock market predictions are inherently uncertain
- Past performance does not guarantee future results
- Always conduct thorough research before investing
- Consider consulting with financial advisors
- Use this tool as one of many inputs for decision-making
- The developers assume no liability for investment decisions

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- New features
- Documentation improvements
- Additional technical indicators
- New regression models

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **Plotly**: For interactive visualizations
- **scikit-learn**: For machine learning models
- **yfinance**: For stock data access
- **ta (Technical Analysis)**: For technical indicators

## 📧 Contact

For questions, suggestions, or issues:

- Open an issue on GitHub
- Submit a pull request
- Contact the maintainers

---

**Happy Predicting! 📈🚀**

_Remember: Always invest responsibly and never invest money you can't afford to lose._
