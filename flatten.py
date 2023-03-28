import cupy as cp
import numpy as np
from Layer import Layer

class Flatten(Layer):
    def forward(self, x,training=True):
        self.input = x

        if type(x) == np.ndarray:
            return np.array([i.flatten() for i in x]).T
        else:
            return cp.array([i.flatten() for i in x]).T

    def backward(self, outputGradient):
        _, depth, width, height = self.input.shape
        
        if type(outputGradient) == np.ndarray:
            return np.array([np.reshape(grad, (depth, width, height)) for grad in outputGradient.T])
        else:
            return cp.array([cp.reshape(grad, (depth, width, height)) for grad in outputGradient.T])



