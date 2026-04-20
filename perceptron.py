import numpy as np
from config import RANDOM_SEED, LEARNING_RATE

# set random seed for reproducible weight initialization
np.random.seed(RANDOM_SEED)

class Perceptron:

    # initialize the perceptron with random weights and bias, and a specified learning rate
    def __init__(self, input_size, learning_rate=LEARNING_RATE):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate
        
        # Pocket Learning Algorithm: store best weights and bias found so far
        self.best_weights = self.weights.copy()
        self.best_bias = self.bias
        self.best_errors = float('inf')
    
    # calculate the raw score for a given input vector x
    def score(self, x):
        return np.dot(self.weights, x) + self.bias
    
    # predict the output for a given input vector x using the current weights and bias
    def predict(self, x):
        if self.score(x) > 0:
            return 1
        else:
            return 0
    
    # count errors on a dataset (used by Pocket Algorithm)
    def count_errors(self, X, y):
        errors = 0
        
        for x, target in zip(X, y):
            prediction = self.predict(x)
            if prediction != target:
                errors += 1

        return errors
    
    # update the weights and bias based on the error between the target output and the predicted output
    # for a given input vector x
    def train(self, x, target):
        prediction = self.predict(x)

        error = target - prediction

        self.weights += self.lr * error * x
        self.bias += self.lr * error
    
    # Pocket Learning Algorithm: update best weights if current ones are better
    def update_pocket(self, X, y):
        current_errors = self.count_errors(X, y)
        
        if current_errors < self.best_errors:
            self.best_errors = current_errors
            self.best_weights = self.weights.copy()
            self.best_bias = self.bias
    
    # restore the best weights found so far
    def restore_best(self):
        self.weights = self.best_weights.copy()
        self.bias = self.best_bias