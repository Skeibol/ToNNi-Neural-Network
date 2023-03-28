from LossFunction import LossFunction
import numpy as np
import cupy as cp

class CCE(LossFunction):
    def __init__(self):
        self.id = "CCE"
        pass

    def loss(self,y_pred,y_true):
        if y_pred.shape[0] == 1:
            raise ValueError("U sur yo dimz gewd?")
        m = y_pred.shape[1]

        if type(y_pred) == np.ndarray:
            self.output =  -np.sum(y_true * np.log(y_pred + 1e-10)) 
        else:
            self.output =  -cp.sum(y_true * cp.log(y_pred + 1e-10))

        return self.output

    def calculateGradient(self,y_pred,y_true):
        if type(y_pred) == cp.ndarray:
            y_true = cp.asarray(y_true)
        m = y_pred.shape[1]
        grad =  y_pred - y_true
        return grad

