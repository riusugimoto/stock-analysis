import numpy as np
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

def scale_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def inverse_transform(scaler, data, original_shape):
    dummies = np.zeros((data.shape[0], original_shape) )
    #multi-dimensional array into a 1D array for further processing.
    #The first column of dummies is assigned the train_predictions
    dummies[:, 0] = data.flatten() 
    #This step reverses the normalization applied earlier, converting the normalized values back to their original scale.
    inversed_data = scaler.inverse_transform(dummies)
    return dc(inversed_data[:, 0])