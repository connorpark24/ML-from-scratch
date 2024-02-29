import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Activation_ReLU:

    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Optimizer_SGD:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        # Update weights and biases according to backpropagation
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    
class Loss_CategoricalCrossEntropy:

    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        # If y_true is not one-hot encoded, we convert it here
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate and normalize gradient
        self.dinputs = dvalues - y_true
        self.dinputs = self.dinputs / samples

        return self.dinputs

class Layer_Dense:
    
    def __init__(self, input_size, output_size):
        # Set weights to be random and biases to be zeros
        self.weights = 0.01 * np.random.rand(input_size, output_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Calculate gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Calculate gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class NeuralNetwork:
    def __init__(self, epochs=10000):
        self.epochs = epochs
        self.optimizer = Optimizer_SGD(0.01)
        self.loss_function = Loss_CategoricalCrossEntropy()
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, X, y):
        for epoch in range(self.epochs):
            # Forward pass
            inputs = X
            for layer in self.layers:
                layer.forward(inputs)
                inputs = layer.output
            output = self.layers[-1].output

            # Calculate loss from output of final layer
            loss = self.loss_function.forward(output, y)

            if epoch % 100 == 0:
                print(f'epoch: {epoch}, loss: {loss}')
            
            # Backward pass
            dvalues = self.loss_function.backward(inputs, y)
            for layer in reversed(self.layers):
                layer.backward(dvalues)
                if hasattr(layer, 'dinputs'):  # Check if the layer has 'dinputs' attribute to continue backward pass
                    dvalues = layer.dinputs

            # Update parameters
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)

    def predict(self):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return self.layers[-1].output

def main():
     # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Initialize the neural network
    nn = NeuralNetwork()
    
    # Add layers
    nn.add_layer(Layer_Dense(2, 64))  # Input layer
    nn.add_layer(Activation_ReLU())
    nn.add_layer(Layer_Dense(64, 3))  # Output layer
    nn.add_layer(Activation_ReLU())

    # Train the neural network
    nn.train(X, y)
    
main()