import numpy as np

class ActivationFunction:
    def __init__(self, function_arg):
        choices = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'softmax': self.softmax
        }

        self.activation_function = choices.get(function_arg, "Invalid activation function")
    
    def relu(x):
        return x if x > 0 else 0

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

class NeuralNetwork:
    def __init__(self, weights, actvation_function):
        self.weights = weights
    
    def feedforward(self, inputs):
        return np.dot(inputs, self.weights)
    
def main():
    weights = np.array([0, 1])
    inputs = np.array([0, 1])
    nn = NeuralNetwork(weights)
    prediction = nn.feedforward(inputs)
    print(prediction)

main()