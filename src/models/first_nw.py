import numpy as np


def neural_network(input, weights):
    pred = input.dot(weights)
    return pred
