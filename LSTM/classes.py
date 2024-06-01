 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from copy import deepcopy as dc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


 ######################################### 
    #Creating a class for the training and test data
    #encapsulates the features (X) and target values (y) in a single object. This makes it easier to manage and access the data. 
class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.y[i]




class LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, 
                            batch_first=True) #input and output tensors are expected to have the shape (batch_size, sequence_length, input_size).
                                              #Without batch_first=True: The default setting. the shape (sequence_length, batch_size, input_size).
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0) #x is the input tensor with shape (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        '''
        out: Contains the output features from the LSTM for each time step. 
        If the input x has the shape (batch_size, sequence_length, input_size), 
        out will have the shape (batch_size, sequence_length, hidden_size).
        '''
        # _: This placeholder captures the new hidden and cell states, which are not used
        out, _ = self.lstm(x, (h0, c0))   
        #fully connectced layer
        out = self.fc(out[:, -1, :]) #The -1 index represents the last element along the second dimension (in this case, the sequence length dimension).
        return out


