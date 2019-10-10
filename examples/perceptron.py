# A small example that learns the OR function. The data is represented as a
# numpy array, and the labels are represented as a vector.

import numpy as np

# Create training data
x_train = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y_train = np.array([-1, 1, 1, 1])
data_train = (x_train, y_train)

# Create a perceptron model
from learnz.ml.models import Perceptron
model = Perceptron()

# Train a model
model.train(data_train, epochs = 100)

# Get an evaluation metric
from learnz.ml.evaluation import accuracy as accuracy

# Make predictions and evaluations on the trained model (we have provided every
# example, so the accuracy should be 1.0 unless we are very unlucky)
[train_accuracy] = model.evaluate(data_train, accuracy)
print(f"Accuracy: {train_accuracy}")
