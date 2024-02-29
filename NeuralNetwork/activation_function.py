import numpy as np

class ActivationFunction:
    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dvalues):
        raise NotImplementedError
    

class Activation_ReLU:

    def __init__(self):
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0