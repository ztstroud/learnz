import numpy as np
from scipy.sparse import csr_matrix

def read_libsvm(data_path, num_features = None):
    """
    Reads a libsvm file to produce features and labels.

    The number of features can be inferred from the data, or specified.

    :param data_path: the path to a libsvm file
    :param num_features: the number of features in the data

    :return: a feature matrix
    :return: a label vector
    """

    with open(data_path, "r") as data_file:
        lines = data_file.readlines()

        if num_features is None:
            num_features = _count_libsvm_features(lines)

        y = []
        row_indices = []
        column_indices = []
        data = []

        for row_index, line in enumerate(lines):
            elements = line.split()
            y.append(float(elements[0]))

            for element in elements[1:]:
                column_index, value = element.split(":")
                
                row_indices.append(row_index)
                column_indices.append(int(column_index))
                data.append(float(value))

        x = csr_matrix((data, (row_indices, column_indices)), shape = (len(y), num_features))

        return x, np.array(y)

def _count_libsvm_features(lines):
    """
    Determine how many featues are present in a libsvm file.

    :param lines: the lines of a libsvm file

    :return: the number of featues in the libsvm file
    """

    max_column_index = -1

    for line in lines:
        elements = line.split()

        for element in elements[1:]:
            column_index = int(element.split(":")[0])

            if column_index > max_column_index:
                max_column_index = column_index

    if max_column_index == -1:
        raise Exception("LibSVM file contains no data")

    return max_column_index + 1
