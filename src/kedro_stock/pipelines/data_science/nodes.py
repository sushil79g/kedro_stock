# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

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



