from learnz.ml.data import read_libsvm

# Read data for using in training and testing
data_train = read_libsvm("learnz/examples/data/perceptron_train")
data_test = read_libsvm("learnz/examples/data/perceptron_test")

from learnz.ml.models import Perceptron
model = Perceptron()

# Define hyperparamter ranges
hyperparameter_ranges = {
    "learning_rate": [1, 0.1, 0.01],
    "averaged": [True, False]
}

from learnz.ml.cross_validation.cross_validation import cross_validate

# Use cross validation to find the best hyperparameters
best_hyperparameters = cross_validate(model, data_train, 5, **hyperparameter_ranges)
print(f"Best Hyperparameters: {best_hyperparameters}")

# Train a model using the best hyperparameters
model.train(data_train, **best_hyperparameters)

# Make predictions and evaluations on the trained model
from learnz.ml.evaluation import accuracy as accuracy
train_predictions, [train_accuracy] = model.predict(data_train, accuracy)
test_predictions, [test_accuracy] = model.predict(data_test, accuracy)

print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
