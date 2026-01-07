import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta

API_KEY = "NCX812PA45NQXGY2"
BASE_URL = "https://www.alphavantage.co/query"


def fetch_stock_data(symbol):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError("Invalid symbol or API limit reached")

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"],
        orient="index"
    )

    df = df.rename(columns={"4. close": "Close"})
    df["Close"] = df["Close"].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    return df


def preprocess_data(df, start_date, end_date):
    df = df.loc[start_date:end_date]

    if df.empty:
        raise ValueError("No data available for selected date range")

    df["Days"] = (df.index - df.index[0]).days
    return df


def predict_future_prices(symbol, start_date, end_date, days_ahead=10):
    df = fetch_stock_data(symbol)
    df = preprocess_data(df, start_date, end_date)

    X = df[["Days"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(
        df["Days"].max() + 1,
        df["Days"].max() + days_ahead + 1
    ).reshape(-1, 1)

    future_prices = model.predict(future_days)

    future_dates = pd.date_range(
        df.index[-1] + timedelta(days=1),
        periods=days_ahead
    ).strftime("%d-%B-%Y").tolist()

    return future_dates, future_prices.tolist()
