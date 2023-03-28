import numpy as np
import skimage
from Layer import Layer

class MaxPooling2D(Layer):
    def __init__(self,kernel_size,stride,padding=0,pool_mode="max"):
        self.kernel_size = kernel_size
        self.noisemap = None
        self.stride = stride
        self.padding = padding
        self.pool_mode = pool_mode
        self.generated = False
        self.initialization = "x"

    def summary(self):
        summary = f"Max Pooling layer, kernel size {self.kernel_size}*{self.kernel_size} , stride: {self.stride}"
        return summary,True

    def forward(self,x,training=True):
        self.input = x
        self.changed = False
        if self.noisemap is None:
            self.noisemap = np.random.normal(0,1e-30,size=x.shape) 

        if not self.noisemap.shape[0]==x.shape[0]:
            self.noisemap = np.random.normal(0,1e-30,size=x.shape) 
            
        self.input = self.input +  self.noisemap #vidi


        if x.shape[2]%2 != 0:
            self.changed = True
            self.input = np.pad(self.input,pad_width=((0,0),(0,0),(0,1),(0,1)),mode="constant",constant_values=0)

        self.output = skimage.measure.block_reduce(self.input, block_size=(1,1, 2, 2), func=np.max)
        self.identity = np.equal(self.input, self.output.repeat(2, axis=2).repeat(2, axis=3)).astype(int)
        #np.sum(self.identity)/self.identity.size mora bude blizu 0.25

        return self.output 
    
    def backward(self,outputGradient):
        if len(outputGradient.shape) != 3: 
            if self.changed:
                outputGradient = outputGradient.repeat(2, axis=2).repeat(2, axis=3) * self.identity #*1/(self.kernel_size*2)
                outputGradient = np.pad(outputGradient,pad_width=((0,0),(0,0),(0,1),(0,1)),mode="constant",constant_values=0)
                outputGradient = outputGradient[:,:,1:-1,1:-1]
            else:
                outputGradient = outputGradient.repeat(2, axis=2).repeat(2, axis=3) * self.identity #*1/(self.kernel_size*2)


            return outputGradient 
    
    def show(self):
        return self.output[0]

    


