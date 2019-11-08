import numpy as np


def evaluate(labels, predictions, *metrics):
    """
    Runs all of the given metrics on the labels and predictions.

    :param labels: the ground truth
    :param predictions: predicted labels
    :param metrics: the metrics to run

    :return: the results of the given metrics
    """

    return [metric(labels, predictions) for metric in metrics]

def accuracy(labels, predictions):
    """
    Determines the accuracy of the given predictions on the given labels.

    :param labels: the ground truth
    :param predictions: predicted labels

    :return: the accuracy of the given predictions on the given labels
    """

    correct = np.equal(labels, predictions)
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

def precision_on(value):
    """
    Returns a function that can be usedto determine the precision of a set of
    predictions on a set of labels. The function should be called with the
    labels as the first parameter and the predictions as the second parameter.

    For example:

        precision_on_one = precision_on(1)
        precision = precision_on_one(labels, predictions)

    :param value: the value to find the precision of

    :return: a function taht determines the precision of given labels and predictions
    """

    return lambda labels, predictions: _precision(value, labels, predictions)

def _precision(value, labels, predictions):
    """
    Determines the precision of the given predictions on the given value.

    Precision is defined as the number of correct guesses out of the number of
    guesses. If 'a' was predicted 100 times, but only 25 of those were correct
    the precision would be 0.25.

    :param value: the value to find the precision of
    :param labels: the ground truth
    :param predictions: predicted labels

    :return: the recall of the predictions for the given value
    """

    correct = np.equal(labels, predictions)
    guesses = np.equal(predictions, value)

    if np.count_nonzero(guesses) == 0:
        return 0

    return np.count_nonzero(correct & guesses) / np.count_nonzero(guesses)

def recall_on(value):
    """
    Returns a function that can be used to determine the recall of a set of
    predictions on a set of labels. The function should be called with the
    labels as the first parameter and the predictions as the second parameter.

    For example:

        recall_on_one = recall_on(1)
        recall = recall_on_one(labels, predictions)

    :param value: the value to find the recall of

    :return: a function that determines the recall of given labels and predictions
    """

    return lambda labels, predictions: _recall(value, labels, predictions)

def _recall(value, labels, predictions):
    """
    Determines the recall of the given labels and predictions on the given value

    :pram value: the value to calculate recall for
    :param labels: the ground truth
    :param predictions: predicted labels

    :return: the recall of the predictions for the given value
    """

    correct = np.equal(labels, predictions)
    with_value = np.equal(labels, value)

    if np.count_nonzero(with_value) == 0:
        return 0

    return np.count_nonzero(correct & with_value) / np.count_nonzero(with_value)
