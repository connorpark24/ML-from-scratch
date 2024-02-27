import numpy as np

class ActivationFunction:
    def __init__(self, function_arg):
        choices = {
            'relu': self.relu,
            'sigmoid': self.sigmoid,
            'tanh': self.tanh,
            'softmax': self.softmax
        }

        self.choices = function_arg
        self.activation_function = choices.get(function_arg, "Invalid activation function")
    
    def relu(x):
        return np.maximum(0, x)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def tanh(x):
        return np.tanh(x)

    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def backward():
        pass

class Optimizer:
    def __init__(self, function_arg, learning_rate):
        choices = {
            'sgd': self.sgd,
            'adam': self.adam,
            'rmsprop': self.rmsprop
        }
        
        self.learning_rate = learning_rate
        self.optimizer = choices.get(function_arg, "Invalid optimizer")

    def update_params(self, layer):
        # Update weights and biases according to backpropagation
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases
    
class LossFunction:
    def __init__(self, function_arg):
        choices = {
            'mean_squared_error': self.mean_squared_error,
            'cross_entropy': self.cross_entropy
        }

        self.loss_function = choices.get(function_arg, "Invalid loss function")
    
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def cross_entropy(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))

class DenseLayer:
    
    def __init__(self, input_size, output_size, activation_function):
        # Set weights to be random and biases to be zeros
        self.weights = 0.01 * np.random.rand(input_size, output_size)
        self.biases = np.zeros((1, output_size))
        self.activation_function = activation_function

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
    def __init__(self, actvation_function, epochs=1000, learning_rate=0.01, optimizer='sgd', loss_function='mean_squared_error'):
        self.activation_function = actvation_function
        self.epochs = epochs
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, data, labels):
        for epoch in range(self.epochs):
            # Forward pass
            for layer in self.layers:
                layer.forward()
            
            # Calculate and print stats
            loss = self.loss_function(self.layers[-1].output, labels)
            accuracy = np.mean(np.argmax(self.layers[-1].output, axis=1) == labels)
            print(f'Epoch: {epoch}, Loss: {loss}, Accuracy: {accuracy}')

            # Backward pass
            for layer in reversed(self.layers):
                layer.backward()
            
            # Update weights and biases
            for layer in self.layers:
                self.optimizer.update_params(layer)        

def main():
    inputs = np.array([0, 1])
    activation_function = ActivationFunction('relu')
    nn = NeuralNetwork(activation_function)
    prediction = nn.feedforward(inputs)
    print(prediction)

main()