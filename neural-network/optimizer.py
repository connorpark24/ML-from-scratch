import numpy as np

class Optimizer:
    def update_params(self, layer):
        raise NotImplementedError

class Optimizer_SGD:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        # Update weights and biases according to backpropagation
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases