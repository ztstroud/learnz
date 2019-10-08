import itertools


def generate_hyperparameters(**kwargs):
    """
    Generates all possible hyperparameter combinations.
    """

    keys, possible_values = zip(*kwargs.items())
    
    for values in itertools.product(*possible_values):
        yield dict(zip(keys, values))
