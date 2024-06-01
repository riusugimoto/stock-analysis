import os
import pandas as pd
import yfinance as yf

def load_sp500_data():

    data_folder = r'C:\Users\agoo1\OneDrive\Documents\2024 summer\Data\mlstock\data'
    os.makedirs(data_folder, exist_ok=True)
    data_path = os.path.join(data_folder, "sp500.csv")

    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv(data_path, index_col=0)
    else:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
    
    sp500.index = pd.to_datetime(sp500.index)
    return sp500

def prepare_data(sp500):
    sp500.index = pd.to_datetime(sp500.index)
    sp500 = sp500.copy()
    del sp500["Dividends"]
    del sp500["Stock Splits"]

    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    sp500 = sp500.loc["1990-01-01":]
    return sp500

