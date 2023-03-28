import numpy as np
from src import *


class Dense:
    def __init__(self,input,output,activation="relu"):
        self.W = np.random.rand(output,input) -0.5
        self.b = np.random.rand(output,1) -0.5
        self.next = None
        self.prev = None
        self.act = activation
        if activation == "relu":
            self.activator = ReLU()
        elif activation == "softmax":
            self.activator = Softmax()
    
    def forward(self):
        
        self.Z1 = self.W.dot(self.prev.output) + self.b 
        self.activator.forward(self.Z1)
        
        self.output = self.activator.output
    def backward(self): 
        m = self.Y.size

        if self.act =="softmax":
            self.next.oneHot(self.Y)
            self.err_Z = self.output - self.next.oneHotY #prev = nxt

        elif self.act=="relu":
            
            self.activator.backward(self.output)
            self.err_Z = self.next.W.T.dot(self.next.err_Z) * self.activator.output # x = prev

        self.err_W = 1/m * self.err_Z.dot(self.prev.output.T) # x = previous
        self.err_b = 1/m * np.sum(self.err_Z)

    def update(self,lr):
        
        self.W = self.W - lr * self.err_W
        self.b = self.b - lr * self.err_b
        
    def flush(self):
        self.err_W = 0
        self.err_b = 0
        self.err_Z = 0
        
def compile(data,Y):
        for i,layer in enumerate(data):
            if i!=0 and i!=len(data) -1:
                layer.prev = data[i-1]
                layer.next = data[i+1]
                layer.Y = Y
                


    

class Input:
    def __init__(self,x):
        self.output = x
       

class Output:
    def __init__(self,x):
        self.output = x

    def oneHot(self,Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        self.oneHotY = one_hot_Y
        

class ReLU:
    def __init__(self) -> None:
        pass
    def forward(self,x):
        self.output = np.maximum(x,0)

    def backward(self,x):
        
        self.output =  x>0
        

class Softmax:
    def __init__(self) -> None:
        pass
    def forward(self,x):
        exp = np.exp(x - np.max(x)) 
        self.output = exp / exp.sum(axis=0)

    def backward(self,x):
        self.output = self.output - x


