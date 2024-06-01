from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas as pd

def train_model(train, predictors):
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    return model


def predict(test, predictors, model):
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    return preds

def predict2(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def predict3(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def evaluate_model(test, preds):
    return precision_score(test, preds)


def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict2(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)


def backtest2(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict3(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)



#rolling https://www.programiz.com/python-programming/pandas/methods/rolling#:~:text=Here%2C%20we%20have%20used%20the,centered%20on%20its%20respective%20window.
def rolling (data):  #data = sp500
    horizons = [2,5,60,250,1000]
    new_predictors = []

    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
        
        ratio_column = f"Close_Ratio_{horizon}" #set a name for this new col ex Close_Ratio_2, Close_Ratio_5
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        
        trend_column = f"Trend_{horizon}" #set a name 
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors+= [ratio_column, trend_column]
        data=data.dropna(subset=data.columns[data.columns != "Tomorrow"])

    return data,  new_predictors
