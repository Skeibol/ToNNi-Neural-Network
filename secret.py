import cupy as cp
import numpy as np
from Layer import Layer
class  gpu(Layer):
    def __init__(self):
        pass

    def summary(self):
        summary = "GPU STARTS HERE!!\n"
        return summary,False
        
    def forward(self,x,training=True):
        self.input = x
        return cp.asarray(x)

    def backward(self,x):
        self.output = x.get()
        return self.output
