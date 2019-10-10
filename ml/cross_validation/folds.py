import copy
import itertools
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

        fold = data[low:high]
        folds.append(fold)

    return folds

def join_folds(folds, holdout_index = None):
    """
    Joins the given folds

    A holdout index can be specified to holdout data.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: joined folds
    """

    if isinstance(folds[0], tuple):
        return _join_folds_tuple(folds, holdout_index)

    if isinstance(folds[0], list):
        return _join_folds_list(folds, holdout_index)

    if isinstance(folds[0], pd.DataFrame):
        return _join_folds_pandas(folds, holdout_index)

    if isinstance(folds[0], np.ndarray):
        return _join_folds_numpy(folds, holdout_index)

    if isinstance(folds[0], scipy.sparse.csr.csr_matrix):
        return _join_folds_csr_matrix(folds, holdout_index)

    raise Exception(f"Unrecognized data type: {typep(data)}")

def _join_folds_tuple(folds, holdout_index = None):
    """
    Joins the given folds

    A holdout index can be specified to holdout data.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: joined folds
    """

    accumulators = [[] for _ in range(len(folds[0]))]
    included_folds = _get_included_folds(folds, holdout_index)

    for fold in included_folds:
        for index in range(len(fold)):
            accumulators[index].append(fold[index])

    joined = [join_folds(x) for x in accumulators]
    return tuple(joined)

def _join_folds_list(folds, holdout_index = None):
    """
    Joins the given folds

    A holdout index can be specified to holdout data.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: joined folds
    """

    included_folds = _get_included_folds(folds, holdout_index)
    return list(itertools.chain(*included_folds))

def _join_folds_pandas(folds, holdout_index = None):
    """
    Joins the given folds

    A holdout index can be specified to holdout data.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: joined folds
    """

    included_folds = _get_included_folds(folds, holdout_index)
    return pd.concat(included_folds)

def _join_folds_numpy(folds, holdout_index = None):
    """
    Joins the given folds

    A holdout index can be specified to holdout data.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: joined folds
    """

    included_folds = _get_included_folds(folds, holdout_index)
    
    # vectors need to be stacked horizontally
    if len(folds[0].shape) == 1:
        return np.hstack(included_folds)

    return np.vstack(included_folds)

def _join_folds_csr_matrix(folds, holdout_index = None):
    """
    Joins the given folds

    A holdout index can be specified to holdout data.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: joined folds
    """

    included_folds = _get_included_folds(folds, holdout_index)
    return scipy.sparse.vstack(included_folds)

def _get_included_folds(folds, holdout_index = None):
    """
    Returns a list that contains the holds to include whe joining.

    If a holdout index is specified, it will not be included.

    :param folds: the folds to join
    :param holdout_index: an index to holdout

    :return: the folds to use when joining
    """

    return [folds[index] for index in range(len(folds)) if index != holdout_index]
