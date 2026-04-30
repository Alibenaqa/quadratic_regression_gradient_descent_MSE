import numpy as np


def mse(y_pred, y_true):
    # Mean squared error
    return np.mean((y_pred - y_true) ** 2)


def rmse(y_pred, y_true):
    # Root of MSE
    return np.sqrt(mse(y_pred, y_true))
