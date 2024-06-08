import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from copy import deepcopy as dc


def getdata(lookback):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    data = pd.read_csv('C:\\Users\\agoo1\\OneDrive\\Documents\\2024_summer\\Data\\stock\\LSTM\\data\\AAPL.csv')
    data = data[['Date', 'Close']]
    # converts the 'Date' column of the DataFrame to datetime objects.
    data['Date'] = pd.to_datetime(data['Date'])  # data.index = pd.to_datetime(data.index)

    #plt.plot(data['Date'], data['Close'])
   # plt.show()

    transformed_data = transformed_data_for_lstm(data, lookback)
    return transformed_data


def transformed_data_for_lstm(data, n_steps):
    data = dc(data)
        
    # sets the 'Date' column as the index of the DataFrame. index represents time points.
    data.set_index('Date', inplace=True)

    for i in range(1, n_steps+1):
            data[f'Close(t-{i})'] = data['Close'].shift(i)
            data.dropna(inplace=True)
            
    return data

    