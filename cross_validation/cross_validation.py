import itertools
import numpy as np

from learnz.cross_validation.folds import create_folds, join_folds
import learnz.evaluation


def cross_validate(model, data, fold_count, **hyperparameter_ranges):
    folds = create_folds(data, fold_count)

    best_hyperparameters = None
    best_evaluation = None

    for hyperparameters in generate_hyperparameters(**hyperparameter_ranges):
        evaluation = evaluate_folds(model, folds, **hyperparameters)

        if best_evaluation is None or evaluation > best_evaluation:
            best_hyperparameters = hyperparameters
            best_evaluation = evaluation

    return best_hyperparameters

def generate_hyperparameters(**hyperparameters):
    """
    Generates all possible hyperparameter combinations.
    """

    keys, possible_values = zip(*hyperparameters.items())
    
    for values in itertools.product(*possible_values):
        yield dict(zip(keys, values))

def evaluate_folds(model, folds, **hyperparameters):
    evaluations = []
    for holdout_index in range(len(folds)):
        data_train = join_folds(folds, holdout_index)
        data_test = folds[holdout_index]

        model.train(data_train, **hyperparameters)
        _, [evaluation] = model.predict(data_test, [learnz.evaluation.accuracy])
        evaluations.append(evaluation)

    return np.average(evaluations)
