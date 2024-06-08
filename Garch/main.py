import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import os


start = datetime(2015, 1, 1)
end = datetime(2020, 6, 10)



data = pd.read_csv('C:\\Users\\agoo1\\OneDrive\\Documents\\2024_summer\\Data\\stock\\Garch\\AMZN.csv')
data['Date'] = pd.to_datetime(data['Date'])  
data = data[(data['Date'] >= start) & (data['Date'] <= end)]
data['return'] = data['Adj Close'].pct_change()
data = data.dropna()

plt.figure(figsize=(10,4))
plt.plot(data['Date'], data['return'])
plt.xlabel('Date', fontsize=16)
plt.ylabel('Pct Return', fontsize=16)
plt.title('AMZN Returns', fontsize=20)
plt.savefig(os.path.join('images', 'train_plot.png'))
plt.close()

plt.figure(figsize=(10, 4))
plot_pacf(data['return'], lags=40, method='ywm') #the Yule-Walker method for computing the partial autocorrelations.
plt.savefig(os.path.join('images', 'pacf.png'))
plt.close()

model = arch_model(data['return'], vol='Garch', p=1, q=1)
res = model.fit()

#All the coefficients in your GARCH(1,1) model have p-values much less than 0.05, 
#indicating that they are statistically significant. 
#This means that each term in the model contributes meaningfully to explaining the variance in the return series.
print(res.summary())



#scale of  data (returns) is very small, which can affect the convergence of the optimizer when estimating the model parameters.
#This is common when dealing with financial returns, as they are often in the range of very small numbers.
#To address this issue, you can rescale your returns data by multiplying it by 100 before fitting the GARCH model.
rolling_predictions = []
test_size = 100

for i in range(test_size):
    # Train the model on the dataset excluding the last 'test_size-i' observations
    #The training set grows with each iteration, adding one more observation from the end
    train = data['return'][:-(test_size-i)]
    train_scaled = train * 100

    model = arch_model(train_scaled, p=1, q=1, rescale=False)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
   
    # Forecast the variance for the next period
    pred = model_fit.forecast(horizon=1)
    # Append the forecasted standard deviation to the list
    rolling_predictions.append(np.sqrt(pred.variance.values[-1, :][0])/100)

# Convert the rolling predictions list to a Pandas Series with appropriate index
rolling_predictions = pd.Series(rolling_predictions, index=data['return'].index[-365:])




# Plotting true returns and predicted volatility
plt.figure(figsize=(10, 4))
true_returns, = plt.plot(data['Date'][-365:], data['return'][-365:], label='True Returns')
predicted_volatility, = plt.plot(data['Date'][-365:], rolling_predictions, label='Predicted Volatility')
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(handles=[true_returns, predicted_volatility], fontsize=16)
plt.xlabel('Date')
plt.ylabel('Value')
plt.savefig(os.path.join('images', 'pred.png'))
plt.close()

# Plotting squared returns vs. predicted volatility for better comparison
plt.figure(figsize=(10, 4))
squared_returns, = plt.plot(data['Date'][-365:], data['return'][-365:]**2, label='Squared Returns')
predicted_volatility, = plt.plot(data['Date'][-365:], rolling_predictions, label='Predicted Volatility')
plt.title('Squared Returns vs. Predicted Volatility', fontsize=20)
plt.legend(handles=[squared_returns, predicted_volatility], fontsize=16)
plt.xlabel('Date')
plt.ylabel('Value')
plt.savefig(os.path.join('images', 'squared_pred.png'))
plt.close()
