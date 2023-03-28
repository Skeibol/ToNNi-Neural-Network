import numpy as np
import cupy as cp
from Layer import Layer

class ReLU(Layer):
    def __init__(self):
        pass

    def forward(self, x, training=True):

        if type(x) == np.ndarray:
            self.input = x
            self.output = np.maximum(x, 0)

        else:
            self.input = x
            self.output = cp.maximum(x, 0)

        return self.output

    def backward(self, outputGradient):
        return outputGradient*(self.output > 0)

    def summary(self):
        summary = "ReLU activation"
        return summary, False


class Softmax(Layer):
    def __init__(self):
        pass

    def forward(self, x, training=True):

        if type(x) == np.ndarray:
            self.input = x
            a = np.exp(x - np.max(x))
            self.output = a / a.sum(axis=0)
        else:
            self.input = x
            exp = cp.exp(x - cp.max(x))
            if exp.shape[1] == 1:
                self.output = exp / exp.sum()
            else:

                self.output = exp / exp.sum(axis=0)

        return self.output

    def backward(self, y_true):
        #computed in lossfunction
        return y_true

    def summary(self):
        summary = "Softmax output activation"
        return summary, False


class Sigmoid(Layer):
    def __init__(self):
        pass

    def forward(self, x, training=True):
        if type(x) == np.ndarray:
            self.input = x
            self.output = 1 / (1 + np.exp(-x))
        else:
            self.input = x
            self.output = 1 / (1 + cp.exp(-x))

        return self.output

    def backward(self, x):
        return x*(self.output * (1 - self.output))

    def summary(self):
        summary = "Sigmoid activation"
        return summary, False


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, training=True):
        if type(x) == np.ndarray:
            self.input = x
            self.output = np.maximum(x, self.alpha * x)
        else:
            self.input = x
            self.output = cp.maximum(x, self.alpha * x)

        return self.output

    def backward(self, outputGradient):
        self.output[self.output > 0] = 1
        self.output[self.output <= 0] = self.alpha
        return outputGradient*self.output

    def summary(self):
        summary = "Leaky ReLU activation"
        return summary, False


class Tanh(Layer):
    def __init__(self):
        pass

    def forward(self, x, training=True):
        if type(x) == np.ndarray:
            self.input = x
            self.output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        else:
            self.input = x
            self.output = (cp.exp(x) - cp.exp(-x)) / (cp.exp(x) + cp.exp(-x))

        return self.output

    def backward(self, x):
        if type(x) == np.ndarray:
            return x * (1 - np.power(self.output, 2))
        else:
            return x * (1 - cp.power(self.output, 2))

    def summary(self):
        summary = "TanH activation"
        return summary, False
