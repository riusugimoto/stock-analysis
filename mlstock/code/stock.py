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
#print(sp500)

sp500.plot.line(y="Close", use_index=True)
# Save the plot as an image file
plt.savefig("sp500_plot.png")
# Close the plot to prevent it from displaying
plt.close()

del sp500["Dividends"]
del sp500 ["Stock Splits"]

sp500["Tomorrow"] = sp500["Close"].shift(-1) 
sp500["Target"]= (sp500["Tomorrow"] > sp500["Close"]).astype(int)
sp500 = sp500.loc["1990-01-01":].copy()
print(sp500)

#Randomforest 
#random_state=1 means setting the seed. get the same result when you run again
model = RandomForestClassifier( n_estimators=100,  min_samples_split=100, random_state=1)
train = sp500.iloc[:-100] #use all the rows except the last 100 rows 
test = sp500.iloc[-100:] #use the last 100 rows
predictors = ["Close", "Volume", "Open", "High", "Low"] # make a list 
model.fit(train[predictors], train["Target"])

#predict and see errors
preds = model.predict(test[predictors])
print(preds)
preds = pd.Series(preds, index=test.index)
print(preds)
print(precision_score(test["Target"], preds))

combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()
plt.savefig("combined_plot.png")
plt.close()

