import numpy as np


def accuracy(labels, predictions):
    """
    Determines the accuracy of the given predictions on the given labels.

    :param labels: the ground truth
    :param predictions: predicted labels

    :return: the accuracy of the given predictions on the given labels
    """

    correct = np.equal(labels, predictions.T)
    correct_count = np.count_nonzero(correct)

    return correct_count / len(labels)

def error(labels, predictions):
    """
    Determines the error of the given predictions on the given labels.

    :param labels: the ground truth
    :param predictions: predicted labels

    :return: the error of the given predictions on the given labels
    """

    return 1 - accuracy(labels, predictions)
