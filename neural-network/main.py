import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from activation_function import Activation_ReLU
from loss import Loss_CategoricalCrossEntropy
from optimizer import Optimizer_SGD

nnfs.init()

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
                dvalues = layer.dinputs

            # Update parameters
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    self.optimizer.update_params(layer)  # Don't optimize on activation layers

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