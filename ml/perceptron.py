import numpy as np
import random


def train(data, *, learning_rate = 1, decay_learning_rate = False, averaged = True, epochs = 10):
    """
    Trains a perceptron using the given data.

    :param data: the data to use in training
    :param learning_rate: the learning rate
    :param decay_learning_rate: whether or not to decay the learning rate with each epoch
    :param averaged: return the average of all weights
    :param epochs: the number of epochs to run
    """

    num_examples, num_features = data[0].shape

    weights = np.array([[random.uniform(-0.01, 0.01) for _ in range(num_features)]])
    total_weights = np.array([[0.0 for _ in range(num_features)]])

    for epoch in range(epochs):
        for example, label in enumerate_data(data):
            prediction = predict(example, weights)

            if prediction != label:
                weights += learning_rate / (1 + epoch) * label * example

            if averaged:
                total_weights += weights

    if averaged:
        return total_weights / (num_examples * epochs)

    return weights

def enumerate_data(data):
    """
    Enumerates the given data in a random order.

    :param data: the data th enumerate
    """

    x, y = data
    
    indices = [index for index in range(len(y))]
    random.shuffle(indices)

    for index in indices:
        yield x[index, :], y[index]

def predict(example, weights):
    """
    Makes a prediction for each of the given examples.

    :param example: the examples to predict
    :param weights: the weights to use in the prediction

    :return: predictions for each example
    """

    return np.sign((example * weights.T))

