from LossFunction import LossFunction
import numpy as np
import cupy as cp

class BCE(LossFunction):
    def __init__(self):
        self.id = "BCE"
        pass

    def loss(self,y_pred,y_true):
        if y_pred.shape[0] != 1:
            raise ValueError("U sur yo dimz gewd?")
        
        m = y_pred.shape[1]
        self.input = y_pred
        
       
        y_true = y_true + 1e-50
        if type(y_pred) == np.ndarray:
            y_true = np.asarray(y_true)    
            y_pred = np.clip(y_pred,1e-17,0.999999999)
            self.output = -((y_true*cp.log(y_pred) + (1-y_true)*cp.log(1-y_pred))) * 1/m
            return self.output
        
        else:
            y_pred = cp.clip(y_pred,1e-17,0.999999999)
            self.output = -((y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))) * 1/m
            return self.output


    def calculateGradient(self,y_pred,y_true):
        if type(y_pred) == np.ndarray:
            y_true = np.asarray(y_true)
            y_pred = np.clip(y_pred,1e-17,0.999999999)
        else:
            y_pred = cp.clip(y_pred,1e-17,0.999999999)
            
        m = y_pred.shape[1]
        y_true = y_true + 1e-50
        grad = y_pred * (1-y_pred) 
        grad = ((y_pred-y_true)/(y_pred*(1-y_pred)))
        return grad * 1/m