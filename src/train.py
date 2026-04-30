import numpy as np

from src.loss import rmse
from src.model import linear_predict, quadratic_predict


def _gradients_linear(a, b, x, y):
    # Gradients: linear model
    n = len(x)
    e = linear_predict(a, b, x) - y
    return (2 / n) * np.sum(e * x), (2 / n) * np.sum(e)


def _gradients_quadratic(a, b, c, x, y):
    # Gradients: quadratic model
    n = len(x)
    e = quadratic_predict(a, b, c, x) - y
    return (
        (2 / n) * np.sum(e * x**2),
        (2 / n) * np.sum(e * x),
        (2 / n) * np.sum(e),
    )


def gradient_descent_linear(x, y, lr=0.1, epochs=1000, seed=42):
    # Train linear model
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(2)
    history = []
    for _ in range(epochs):
        da, db = _gradients_linear(a, b, x, y)
        a -= lr * da
        b -= lr * db
        history.append(rmse(linear_predict(a, b, x), y))
    return a, b, history


def gradient_descent_quadratic(x, y, lr=0.1, epochs=1000, seed=42):
    # Train quadratic model
    rng = np.random.default_rng(seed)
    a, b, c = rng.standard_normal(3)
    history = []
    for _ in range(epochs):
        da, db, dc = _gradients_quadratic(a, b, c, x, y)
        a -= lr * da
        b -= lr * db
        c -= lr * dc
        history.append(rmse(quadratic_predict(a, b, c, x), y))
    return a, b, c, history
