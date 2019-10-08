import copy
import pandas as pd
import random


def create_folds(data, count):
    """
    Creates folds from the given data.

    :param data: the data to fold
    :param count: the number of folds to create

    :return: a list of folds
    """

    if count < 1:
        raise Exception("You cannot have less than one fold")

    if count > len(data):
        raise Exception("You cannot have more folds than examples")

    if isinstance(data, list):
        return _create_folds_list(data, count)

    if isinstance(data, pd.DataFrame):
        return _create_folds_pandas(data, count)

    raise Exception(f"Unrecognized data type: {type(data)}")

def _create_folds_list(data, count):
    """
    Creates folds from the given data.

    :param data: the data to fold
    :param count: the number of folds to create

    :return: a list of folds
    """

    fold_count = len(data) / count
    folds = list()

    for fold_index in range(count):
        low = int(fold_index * fold_count)
        high = int((fold_index + 1) * fold_count)

        fold = data[low:high]
        folds.append(fold)

    return folds

def _create_folds_pandas(data, count):
    """
    Creates a fold from the given data

    :param data: the data to fold
    :param count: the number of folds to create

    :return: a list of folds
    """

    fold_count = len(data) / count
    folds = list()

    for fold_index in range(count):
        low = int(fold_index * fold_count)
        high = int((fold_index + 1) * fold_count)

        fold = data.iloc[low:high]
        folds.append(fold)

    return folds
