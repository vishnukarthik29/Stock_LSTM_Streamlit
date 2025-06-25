# from nsepy import get_history
# from datetime import date
# import pandas as pd

# # Download data for TCS from NSE
# df = get_history(
#     symbol="TCS",                    # No ".NS"
#     start=date(2010, 1, 1),          # Start date
#     end=date.today(),               # End date (today)
# )

# print(df.shape)
# print(df[['Close']].tail())  # Display last 5 closing prices
import yfinance as yf
print(yf.download("TCS.NS", period="1y").head())
df = yf.download("TCS.BO", period="5y", interval="1d")
df = yf.download("AAPL", period="5y", interval="1d")
# import yfinance as yf

# tickers = ["AAPL", "TCS.NS", "RELIANCE.NS"]

# for ticker in tickers:
#     print(f"\n🔍 Checking: {ticker}")
#     try:
#         df = yf.download(ticker, period="1y", interval="1d", progress=False)
#         if df.empty:
#             print("❌ No data found.")
#         else:
#             print("✅ Data fetched successfully!")
#             print(df.tail())
#     except Exception as e:
#         print(f"❌ Error for {ticker} -> {e}")
# from alpha_vantage.timeseries import TimeSeries
# import pandas as pd

# api_key = '70CILADGUDA5YE75'  # <-- paste your API key here
# ts = TimeSeries(key=api_key, output_format='pandas')

# symbols = ['RELIANCE.BSE', 'TCS.BSE']  # For Indian stocks

# for symbol in symbols:
#     print(f"\n🔍 Fetching: {symbol}")
#     try:
#         data, meta = ts.get_daily(symbol=symbol, outputsize='compact')
#         print("✅ Data fetched successfully!")
#         print(data.tail())
#     except Exception as e:
#         print(f"❌ Failed to fetch data: {e}")

