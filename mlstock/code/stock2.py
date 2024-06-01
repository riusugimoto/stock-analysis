import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")

sp500.index = pd.to_datetime(sp500.index)

del sp500["Dividends"]
del sp500 ["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1) 
sp500["Target"]= (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
print(sp500)

model = RandomForestClassifier( n_estimators=100,  min_samples_split=100, random_state=1)
train = sp500.iloc[:-100] #use all the rows except the last 100 rows 
test = sp500.iloc[-100:] #use the last 100 rows
predictors = ["Close", "Volume", "Open", "High", "Low"] # make a list 


def predct (train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions =[]

    for i in range (start, data.shape[0], step):
        train = data.iloc[:i]
        test = data.iloc[i:i+step]
        predictions = predct(train, test, predictors, model)
        all_predictions.append(predictions)

    return  pd.concat(all_predictions)


predictions = backtest(sp500, model, predictors)
print(predictions)
print(precision_score(predictions["Target"], predictions["Predictions"]))

