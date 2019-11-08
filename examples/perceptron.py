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
from learnz.ml.evaluation import accuracy, precision_on, recall_on
import learnz.ml.evaluation as evaluation

# Make predictions and evaluations on the trained model (we have provided every
# example, so the accuracy should be 1.0 unless we are very unlucky)
predictions, [train_accuracy, train_precision, train_recall] = model.predict(data_train, accuracy, precision_on(1), recall_on(1))

print(f"Ground Truth:\n  {y_train}\n")
print(f"Predictions:\n  {predictions}\n")

print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
