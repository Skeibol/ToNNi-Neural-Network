from LossFunction import LossFunction
import numpy as np
import numpy as np

class MSE(LossFunction):
    def __init__(self):
        self.id = "MSE"

    def loss(self, y_pred, y_true):
        # y = 2*x + 100
        # h = a*x + b

        m = y_pred.shape[1] # m = 1
        self.output = 0
        
        if type(y_pred) == np.ndarray: # #vektorizirat
            for i in range(m):
                self.output += ((y_pred - y_true) * (y_pred - y_true))
        else:
            for i in range(m):
                self.output += ((y_pred - y_true) * (y_pred - y_true))

        self.output /= (2 * m)
        return self.output #/ (2 * m)

    def calculateGradient(self,y_pred,y_true):
        if type(y_pred) == np.ndarray:
            y_true = np.asarray(y_true)
        
        m = y_pred.shape[1] # m=1; m je batchsize?xd
        
        grad = 0 #vektorizirat
        for i in range(m):
            grad += (y_pred - y_true) # * self.input) # p = 1 ili x #mozda i ne ovako; u x-u sta.. x i 1? ne

        return grad / m # jel triba *x, *self.output? /m?
