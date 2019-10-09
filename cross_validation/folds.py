import copy
import numpy as np
import pandas as pd
import random
import scipy.sparse.csr


def create_folds(data, count):
    """
    Creates folds from the given data.

    :param data: the data to fold
    :param count: the number of folds to create

    :return: a list of folds
    """

    if count < 1:
        raise Exception("You cannot have less than one fold")

    if isinstance(data, tuple):
        return _create_folds_tuple(data, count)

    if isinstance(data, list):
        return _create_folds_list(data, count)

    if isinstance(data, pd.DataFrame):
        return _create_folds_pandas(data, count)

    if isinstance(data, np.ndarray) or isinstance(data, scipy.sparse.csr.csr_matrix):
        return _create_folds_numpy(data, count)

    raise Exception(f"Unrecognized data type: {type(data)}")

def _create_folds_tuple(data, count):
    """
    Create folds from the given data.

    Folds will be created from each element of the tuple, and returned as a list
    of tuples.

    :param data: the data to fold
    :param count: the number of folds to create

    :return: a list of folds
    """

    subfolds = [create_folds(subdata, count) for subdata in data]
    return [fold for fold in zip(*subfolds)]

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
    Creates folds from the given data

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

def _create_folds_numpy(data, count):
    """
    Creates folds from the given data

    :param data: the data to fold
    :param count: the number of folds to create

    :return: a list of folds
    """

    fold_count = data.shape[0] / count
    folds = list()

    for fold_index in range(count):
        low = int(fold_index * fold_count)
        high = int((fold_index + 1) * fold_count)

        fold = data[low:high, :]
        folds.append(fold)

    return folds
