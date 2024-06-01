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

from data_load import getdata
from classes import TimeSeriesDataset
from classes import LSTM

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    lookback = 7
    data= getdata(lookback)
    data_as_np = data.to_numpy()


    # normalized data can speed up the convergence of the learning algorithm and lead to better performance.
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalided_data_as_np = scaler.fit_transform(data_as_np)

    X = normalided_data_as_np[:, 1:] # all rows and all cols except first one(starting from 2nd col)  col-1, col-2....col-7
    y = normalided_data_as_np[:, 0]  # all rows and first col   predicting col


#########################################
 #reverses the order of elements along the specified axis axis=1 refers to the columns. Therefore, np.flip(X, axis=1) reverses the order of columns for each row.
 #LSTMs and other recurrent neural networks typically process sequences from past to present. By flipping the sequence, 
 #the data starts with the oldest lag and ends with the most recent, aligning with how these models are usually structured to process time series data.
    X = dc(np.flip(X, axis=1)) # col-1, col-2...col-7 --> col-7....., col-2,col-1

    split_index = int(len(X) * 0.95)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]
    

    #LSTM networks expect their input data to be in a specific three-dimensional shape: (number of samples, number of time steps, number of features).
    #number of samples, lookback, number of features
    #-1 is used to automatically calculate the number of samples based on the length of X_train 
    #lookback is the number of time steps (lags) used as input features.
    #1 indicates that there is only one feature per time step (the closing price).

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    # (number of samples, 1)  1 means just using 1 feature, the closing price  
    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

 
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)


# manages batching, shuffling, and parallel data loading, which are essential for:
# Efficient use of computational resources.
# Managing memory effectively.
# Improving model training and generalization.
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#creating the model
    model = LSTM(1, 4, 1)
    model.to(device)

    learning_rate = 0.001
    num_epochs = 10
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_one_epoch(epoch, model, train_loader, loss_function, optimizer)
        validate_one_epoch(model, test_loader, loss_function)

    with torch.no_grad():
        predicted = model(X_train.to(device)).to('cpu').numpy()

    plt.plot(y_train, label='Actual Close')
    plt.plot(predicted, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()



#training the model.
def train_one_epoch(epoch, model, train_loader, loss_function, optimizer):
        model.train(True) # Sets the model to training mode. 
        print(f'Epoch: {epoch + 1}')
        running_loss = 0.0
        
        #training loop
        #iterate over train_loader using enumerate to get both the batch index and the batch data.
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device) # input features and target features
            
            #Forward Pass
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
            #Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 100 == 99:  # print every 100 batches
                avg_loss_across_batches = running_loss / 100
                print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                        avg_loss_across_batches))
                running_loss = 0.0
        print()

#testing the model.
def validate_one_epoch(model, test_loader, loss_function):
    model.train(False) # Set the model to evaluation mode
    running_loss = 0.0
        
    for batch_index, batch in enumerate(test_loader):

        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
                    
        #No Gradient Calculation,disables gradient calculation since no needto calculate gradients because you are not updating the model's parameters
        with torch.no_grad():
            output = model(x_batch) #foward pass
            loss = loss_function(output, y_batch) # Computes the loss 
            running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)
        
        print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
        print('***************************************************')
        print()












if __name__ == "__main__":
    main()

