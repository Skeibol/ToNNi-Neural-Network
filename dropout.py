import numpy as np
import cupy as cp
from Layer import Layer

class Dropout(Layer):
    def __init__(self,threshold=0.8):
        self.threshold = threshold

    def summary(self):
        summary = "Dropout regularization"
        return summary,False

    def forward(self,x,training=True):
        if not training: return x

        if type(x) == cp.ndarray:

            self.binaryMask = cp.random.rand(*x.shape) < self.threshold
            res = cp.multiply(x, self.binaryMask)
            res *= (1/self.threshold)  # this line is called inverted dropout technique
            return res

        else:
            
            self.binaryMask = np.random.rand(*x.shape) < self.threshold
            res = np.multiply(x, self.binaryMask)
            res *= (1/self.threshold)  # this line is called inverted dropout technique
            return res

    def backward(self,x):
        x = x * self.binaryMask
        x *=  (1/self.threshold)
        return x
