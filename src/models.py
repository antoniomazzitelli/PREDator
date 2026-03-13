import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM, Dense, Dropout


class RNN:
    """
    Stacked LSTM architecture for multi-step time series forecasting.
    Two LSTM layers with dropout, followed by a Dense output of size forecast_horizon.
    """
    def __init__(self, x_train, y_train, n_epochs, forecast_horizon, n_neurons, batchsize=24): 
        self.x_train = x_train 
        self.y_train = y_train
        self.forecast_horizon = forecast_horizon 
        self.n_neurons = n_neurons 
        
    def RNN(self, callbacks=None):
        """Builds and compiles the model. Returns the unfit Keras Sequential instance."""
        model = Sequential() 
        model.add(LSTM(self.n_neurons, 
                       return_sequences=True, 
                       input_shape=(self.x_train.shape[1], self.x_train.shape[2]))) 
        model.add(Dropout(0.2)) 
        model.add(LSTM(self.n_neurons, return_sequences=False)) 
        model.add(Dropout(0.2)) 
        model.add(Dense(self.forecast_horizon)) 
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        return model
