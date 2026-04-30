import numpy as np


def linear_predict(a, b, x):
    # Linear: ax + b
    return a * x + b


def quadratic_predict(a, b, c, x):
    # Quadratic: ax² + bx + c
    return a * x**2 + b * x + c
