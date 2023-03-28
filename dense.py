import numpy as np
import cupy as cp
from Layer import Layer
from adam import AdamOptim
from SGD import SGD
class Dense(Layer):
    def __init__(self,output,initialization="he_standard",seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.output_size = output
        self.initialized = False
        self.optimizer = None
        self.initialization = initialization.lower().split("_")


    def forward(self,x,training=True):
        if not training: self.t = 1
        if not self.initialized: self.initializeLayer(x)

        self.input = x
        self.output = self.W.dot(self.input) + self.b 

        return self.output

    def backward(self,outputGradient): 
        m1 = outputGradient.size

        self.dW =  outputGradient.dot(self.input.T) * 1/m1
        self.dB = outputGradient.sum(axis=1,keepdims=True) * 1/m1
        self.inputGradient = self.W.T.dot(outputGradient)

        return self.inputGradient
     
    def update(self,lr,optimizer):
        if self.optimizer == None:
            if optimizer.lower() == "adam":
                self.optimizer = AdamOptim()
            if optimizer.lower() == "sgd":
                self.optimizer = SGD()

        self.W,self.b = self.optimizer.optimize(self.t,lr,self.dW,self.dB,self.W,self.b)
        self.t += 1

    def initializeLayer(self,x):
        self.t = 1
        output = self.output_size
        input = x.shape[0]

        if self.initialization[0] == "he":
            self.stdDev = np.sqrt(2/input)
        elif self.initialization[0] == "xavier":
            if self.initialization[1] == "uniform":
                self.stdDev = np.sqrt(6.0/(input+output))
            else:
                self.stdDev = np.sqrt(2.0/(input+output))

        elif self.initialization[0] == "abram":
            self.stdDev = np.sqrt(1/input)

        elif self.initialization[0] == "default":
            self.stdDev = 1.0
        elif self.initialization[0] == "zero":
            self.stdDev = 0.0

        if type(x) == np.ndarray:
            if self.initialization[1] == "standard":
                self.W = np.random.normal(0,self.stdDev,size=(output,input))
            elif self.initialization[1] == "uniform":
                self.W = np.random.uniform(0,self.stdDev,size=(output,input))
            self.b = np.zeros((output,1)) 
            
        else:
            if self.initialization[1] == "standard":
                self.W = cp.random.normal(0,self.stdDev,size=(output,input))
            elif self.initialization[1] == "uniform":
                self.W = cp.random.uniform(0,self.stdDev,size=(output,input))
            self.b = cp.zeros((output,1)) 


        self.initialized = True
    
    def summary(self):
        summary = f"Dense layer , output: {self.output_size}"
        return summary,True
    
    def getParams(self):
        return self.W.size + self.b.size
    
    def getWeights(self):
        return self.W , self.b

    def getStandardDeviation(self):
        if type(self.W) == np.ndarray:
            outputDeviation = np.std(self.output) 
            weightsDeviation = np.std(self.W) - self.stdDev
            return outputDeviation, weightsDeviation
        else:
            outputDeviation = cp.std(self.output) 
            weightsDeviation = cp.std(self.W) - self.stdDev
            return outputDeviation, weightsDeviation