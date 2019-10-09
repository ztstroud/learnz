import math
import pandas as pd


def entropy(data, feature_name):
    """
    Calculates the entropy of the feature with the given name.

    :param data: the data to analyze
    :param feature_name: the name of the feature to analyze

    :return: the entropy of the specified feature
    """

    entropy = 0
    for value_count in data[feature_name].value_counts():
        probability = value_count / len(data)
        entropy -= probability * math.log2(probability) if probability != 0 else 0

    return entropy

def gini_index(data, feature_name):
    """
    Calculates the gini index of the feature with the given name.

    :param data: the data to analyze
    :param feature_name: the name of the feature to anlyze
    
    :return: the gini index of the specified feature
    """

    gini_index = 1
    for value_count in data[feature_name].value_counts():
        probability = value_count / len(data)
        gini_index -= probability * probability

    return gini_index

def information_gain(data, split_feature_name, label_feature_name, metric):
    """
    Calculate the information gain using the given metric.

    :param data: the data to anlyze
    :param split_feature_name: the name of the split feature
    :param label_feature_name: the name of the label feature
    :param metric: the metric to use

    :return: the information gain given by splitting on the split feature
    """

    feature_values = data[split_feature_name]
    possible_values = feature_values.unique()

    expected_metric_value = 0
    for possible_value in possible_values:
        subset = data.loc[data[split_feature_name] == possible_value]

        weight = len(subset) / len(data)
        expected_metric_value += weight * metric(subset, label_feature_name)

    return metric(data, label_feature_name) - expected_metric_value

def get_best_split_feature_name(data, available_feature_names, label_feature_name, metric):
    """
    Calculate the feature that will maximize the information gain if used for a
    split.

    :param data: the data to analyze
    :param available_feature_names: the features available to split on
    :param label_feature_name: the name of the label feature
    :param metric: the metric to use when calculating information gain

    :return: the name of the feature that best splits the data
    """

    best_information_gain = None
    best_split_feature_name = None

    for split_feature_name in available_feature_names:
        feature_information_gain = information_gain(data, split_feature_name, label_feature_name, metric)

        if best_information_gain is None or feature_information_gain > best_information_gain:
            best_information_gain = feature_information_gain
            best_split_feature_name = split_feature_name

    return best_split_feature_name

def id3(data, available_feature_names, label_feature_name, max_depth, metric):
    """
    Runs the ID3 algorithm on the given data.

    :param data: the data to analyze
    :param available_feature_names: the names of features that are available for decisions
    :param label_feature_name: the name of the label feature
    :param max_depth: the maximum depth of the tree
    :param metric: the metric to use when calculating information gain

    :return: the root of a decision tree
    """

    # if a feature only has one unique value we should ignore it because it
    # cannot give us any more information
    available_feature_names = set(feature for feature in available_feature_names if len(data[feature].unique()) > 1)

    # limit the depth of the tree
    if max_depth == 0:
        return LabelNode(data[label_feature_name].mode()[0])

    # if the data only has one label we don't need to make any more decisions
    if len(data[label_feature_name].unique()) <= 1:
        return LabelNode(data.iloc[0][label_feature_name])

    split_feature_name = get_best_split_feature_name(data, available_feature_names, label_feature_name, metric)
    remaining_feature_names = set(feature for feature in available_feature_names if feature != split_feature_name)

    most_common_label = data[label_feature_name].mode()[0]
    root = DecisionNode(split_feature_name, most_common_label)

    for feature_value in data[split_feature_name].unique():
        subset = data[data[split_feature_name] == feature_value]
        root.decisions[feature_value] = id3(subset, remaining_feature_names, label_feature_name, max_depth - 1, metric)

    return root


class DecisionTree:
    def __init__(self, label_name):
        self.label_name = label_name

    def train(self, data, *, max_depth = -1, metric = entropy):
        available_feature_names = {feature_name for feature_name in data.columns if feature_name != self.label_name}
        self._root = id3(data, available_feature_names, self.label_name, max_depth, metric)

    def predict(self, data):
        return pd.Series(self.classify(row) for _, row in data.iterrows())

    def classify(self, example):
        return self._classify(example, self._root)

    def _classify(self, example, tree):
        if isinstance(tree, LabelNode):
            return tree.label

        feature_value = example[tree.feature_name]
        if feature_value in tree.decisions:
            return self._classify(example, tree.decisions[feature_value])

        # if the value is not found, return the most common label
        return tree.most_common_label


class DecisionNode:
    def __init__(self, feature_name, most_common_label):
        self.feature_name = feature_name
        self.most_common_label = most_common_label

        self.decisions = dict()


class LabelNode:
    def __init__(self, label):
        self.label = label
