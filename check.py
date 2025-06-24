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
# print(yf.download("TCS.NS", period="1y").head())
# df = yf.download("TCS.BO", period="5y", interval="1d")
df = yf.download("AAPL", period="5y", interval="1d")
