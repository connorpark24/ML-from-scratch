import numpy as np

class Loss:
    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, dvalues, y_true):
        raise NotImplementedError
    
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