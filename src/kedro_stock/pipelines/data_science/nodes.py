import logging
from typing import Any, Dict

import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
from pandas.io import feather_format

import torch
from torch.autograd import Variable


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

from .model import train_mlmodel

def pre_process(training_df, parameters):
    sc = MinMaxScaler(feature_range = (0,1))
    scaled_training = sc.fit_transform(training_df)
    x_train, y_train = [], []

    for i in range(parameters['INPUT_SIZE'], 1258):
        # import pdb; pdb.set_trace()
        x_train.append(scaled_training[i-parameters['INPUT_SIZE']:i, 0])
        y_train.append(scaled_training[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    
    return x_train, y_train, sc



def train_model(
    train_x: pd.DataFrame, train_y: pd.DataFrame, parameters: Dict[str, Any]
) -> np.ndarray:
    """Node for training a simple multi-class logistic regression model. The
    number of training iterations as well as the learning rate are taken from
    conf/project/parameters.yml. All of the data as well as the parameters
    will be provided to this function at the time of execution.
    """
    x_train, y_train, _ = pre_process(training_df=train_x, parameters=parameters)
    mlmodel = train_mlmodel(x_train, y_train, parameters)

    

    return mlmodel
    
    


def predict(mlmodel: np.ndarray, dataset_total, len_test, training_df,parameters) -> np.ndarray:
    """Node for making predictions given a pre-trained model and a test set.
    """
    inputs = dataset_total[len(dataset_total) - len_test - parameters['INPUT_SIZE']:].values
    inputs = inputs.reshape(-1,1)
    x_train, _, sc = pre_process(training_df, parameters)
    inputs = sc.transform(inputs)
    x_test = []
    for i in range(parameters['INPUT_SIZE'], 80):
        x_test.append(inputs[i-parameters["INPUT_SIZE"]:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    
    x_train_test = np.concatenate((x_train, x_test), axis=0)
    hidden_state = None
    test_input = Variable(torch.from_numpy(x_train_test).float())
    predicted_stock_price, b = mlmodel(test_input, hidden_state)
    predicted_stock_price = np.reshape(predicted_stock_price.detach().numpy(), (test_input.shape[0], 1))
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    return predicted_stock_price
    


def report_metrics(predicted_stock_price: np.ndarray, train_df, test_df, parameters) -> None:
    """Node for reporting the metrices of the predictions performed by the
    previous node. Notice that this function has no outputs, except logging.
    """
    # train_df => training_set
    # test_df = >real_stock_price
    log = logging.getLogger(__name__)
    log.info("started model evaluation")
    real_price = np.concatenate((train_df[parameters['INPUT_SIZE']:], test_df))
    r2_loss = r2_score(real_price, predicted_stock_price)
    mse = mean_squared_error(real_price, predicted_stock_price)
    print("R2 loss of model is", r2_loss)
    print("mean squared error is", mse)



