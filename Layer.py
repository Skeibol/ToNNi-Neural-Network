class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.initialized = False 
        
    def forward(self,input,training=True):
        pass

    def backward(self,input):
        pass

    def update(self,input,optim):
        pass
    
    def summary(self):
        pass

    def checkGradient(self):
        pass

    def getGradients(self):
        pass
    
    def getWeights(self):
        pass
    def flush(self):
        pass

    def show(self):
        pass
    
    def reset(self):
        pass
    def getParams(self):
        pass

    def getStandardDeviation(self):
        pass

    def moveParams(self,minmax):
        pass