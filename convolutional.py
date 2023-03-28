import numpy as np
from Layer import Layer
from adam import AdamOptim
from SGD import SGD
import torch
from torch.nn import functional

class Convolutional2D(Layer):
    def __init__(self, kernel_size, depth,initialization="he_standard",useBias=True,first=False,seed = None):
        if seed is not None:
            np.random.seed(seed)
        self.initialization = initialization.lower().split("_")
        self.initialized = False
        self.depth = depth
        self.kernel_size = kernel_size
        self.optimizer = None
        self.first = first
        self.useBias = useBias

    def summary(self):
        summary = f"Convolutional layer, output: {self.depth} feature maps. Kernel size: {self.kernel_size}*{self.kernel_size}, Using biases: {self.useBias}"
        return summary,True
    
    

    def forward(self, x,training=True):
        if not training: self.t = 1
        if not self.initialized: self.initializeLayers(x)
   
        if type(x) == np.ndarray:
            self.input = torch.from_numpy(x).double()
        if type(self.biases) == np.ndarray:
            self.biases = torch.from_numpy(self.biases).double()
            self.kernels = torch.from_numpy(self.kernels).double()
        if self.useBias:
            
            self.output = functional.conv2d(self.input, self.kernels,bias=self.biases) 
        else:
            self.output = functional.conv2d(self.input, self.kernels) 

        return np.array(self.output)
    
    def backward(self, outputGradient):
        
        m1 = outputGradient.size
        
        if self.useBias:
            self.biasGradient = outputGradient.sum(axis=0).sum(axis=1).sum(axis=1) * 1/m1

        _input = np.array(self.input)
        _input = np.reshape(_input,(_input.shape[1],-1,_input.shape[2],_input.shape[3]))
        _input = torch.from_numpy(_input)
        
        _outputGradient = outputGradient.reshape((outputGradient.shape[1],-1,outputGradient.shape[2],outputGradient.shape[2]))
        _outputGradient = torch.from_numpy(_outputGradient).double()
        
        self.kernelsGradient = functional.conv2d(_input, _outputGradient,padding="valid") 
        self.kernelsGradient = np.array(self.kernelsGradient)  * 1/m1
        self.kernelsGradient = np.reshape(self.kernelsGradient,self.kernels_shape)

        if type(outputGradient) == np.ndarray:
            outputGradient = np.pad(outputGradient,((0,0),(0,0),(2,2),(2,2))) 
            outputGradient =  torch.from_numpy(outputGradient).double()

        _kernels = np.reshape(np.array(self.kernels)[:,:,::-1,::-1],(self.kernels.shape[1],-1,3,3))
        _kernels = torch.from_numpy(_kernels.copy())
        self.inputGradient = functional.conv2d(outputGradient,_kernels,padding="valid")
                 
                
        return np.array(self.inputGradient)
    
    def update(self,lr,optimizer):
        #self.kernelsGradient = np.clip(self.kernelsGradient,-1,1)
        #self.biasGradient = np.clip(self.biasGradient,-1,1)
        if self.optimizer == None:
            if optimizer.lower() == "adam":
                self.optimizer = AdamOptim()
            if optimizer.lower() == "sgd":
                self.optimizer = SGD()

        self.kernels,self.biases = self.optimizer.optimize(self.t,lr,self.kernelsGradient,self.biasGradient,self.kernels,self.biases)  
        self.t += 1

    def initializeLayers(self,x):
        self.t = 1
        input_depth, input_height, input_width = (x.shape[1],x.shape[2],x.shape[3])
        self.input_depth = input_depth
        self.output_shape = (x.shape[0],self.depth,input_height,input_width)
        self.kernels_shape = (self.depth, input_depth, self.kernel_size, self.kernel_size)

        if self.initialization[0] == "he":
            self.stdDev = np.sqrt(2/np.prod(x.shape))
        elif self.initialization[0] == "xavier":
            if self.initialization[1] == "uniform":
                self.stdDev = np.sqrt(6.0/(np.prod(x.shape)+np.prod(self.output_shape)))
            else:
                self.stdDev = np.sqrt(2.0/(np.prod(x.shape)+np.prod(self.output_shape)))
        elif self.initialization[0] == "abram":
            self.stdDev = np.sqrt(1/np.prod(x.shape))
        elif self.initialization[0] == "default":
            self.stdDev = 1.0
        elif self.initialization[0] == "zero":
            self.stdDev = 0.0

        if self.initialization[1] == "standard":
            self.kernels = np.random.normal(0,self.stdDev,size=(self.kernels_shape)).astype(np.float32)   
        elif self.initialization[1] == "uniform":
            self.kernels = np.random.uniform(0,self.stdDev,size=(self.kernels_shape)).astype(np.float32) 
        
        self.biases = np.zeros((self.depth),dtype=np.float32)   
        
        self.biasGradient = np.zeros_like(self.biases).astype(np.float32)
        self.kernelsGradient = np.zeros_like(self.kernels).astype(np.float32)
        self.initialized = True
    
    def show(self):
        return self.output[0]

    def getParams(self):

        return np.prod(self.kernels.size())+ self.biases.shape[0]

    def getStandardDeviation(self):
        outputDeviation = np.std(self.output)
        weightsDeviation = abs(np.std(self.kernels) - self.stdDev)
        return outputDeviation, weightsDeviation 

