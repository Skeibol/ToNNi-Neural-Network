import numpy as np
import cupy as cp
from Layer import Layer
from adam import AdamOptim
from SGD import SGD

class Batchnorm(Layer):
    def __init__(self):
        self.initialized = False
        self.optimizer = None

    def forward(self, x, training=True):
        if not self.initialized:
            self.initializeLayer(x)
        if not training:
            self.t = 1
            # return X

        self.xshape = x.shape
        if len(x.shape) == 2:
            self.X_flat = x.reshape(x.shape[1], -1)
        elif len(x.shape) == 4:
            self.X_flat = x.ravel().reshape(x.shape[0], -1)

        if self.X_flat.shape[0] == 1:
            self.mu = np.mean(self.X_flat)
            self.var = np.var(self.X_flat)
            if type(x) == np.ndarray:
                self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8)
            else:
                self.X_norm = (self.X_flat - self.mu)/cp.sqrt(self.var + 1e-8)
            #self.X_norm = self.X_flat
            out = self.gamma * self.X_norm + self.beta
            return out.reshape(self.xshape)

        else:
            if type(x) == np.ndarray:
                self.mu = np.mean(self.X_flat, axis=0)
                self.var = np.var(self.X_flat, axis=0)
                self.X_norm = (self.X_flat - self.mu)/np.sqrt(self.var + 1e-8)
            else:
                self.mu = cp.mean(self.X_flat, axis=0)
                self.var = cp.var(self.X_flat, axis=0)
                self.X_norm = (self.X_flat - self.mu)/cp.sqrt(self.var + 1e-8)
            out = self.gamma * self.X_norm + self.beta
            return out.reshape(self.xshape)

    def backward(self, outputGradient):

        m = float(np.prod(outputGradient.shape))
        if type(outputGradient) == np.ndarray:
            if len(outputGradient.shape) == 2:
                outputGradient = outputGradient.reshape(outputGradient.shape[1], -1)
            elif len(outputGradient.shape) == 4:
                outputGradient = outputGradient.ravel().reshape(outputGradient.shape[0], -1)

            X_mu = self.X_flat - self.mu
            var_inv = 1./np.sqrt(self.var + 1e-8)

            self.dbeta = np.sum(outputGradient, axis=0) * 1/m
            self.dgamma = outputGradient * self.X_norm * 1/m
            dX_norm = outputGradient * self.gamma
            dvar = np.sum(dX_norm * X_mu, axis=0) * - \
                0.5 * (self.var + 1e-8)**(-3/2)
            dmu = np.sum(dX_norm * -var_inv, axis=0) + dvar * \
                1/self.n_X * np.sum(-2. * X_mu, axis=0)
            dX = (dX_norm * var_inv) + (dmu / self.n_X) + \
                (dvar * 2/self.n_X * X_mu)

            dX = dX.reshape(self.xshape)

        else:
            if len(outputGradient.shape) == 2:
                outputGradient = outputGradient.reshape(outputGradient.shape[1], -1)
            elif len(outputGradient.shape) == 4:
                outputGradient = outputGradient.ravel().reshape(outputGradient.shape[0], -1)

            X_mu = self.X_flat - self.mu
            var_inv = 1./cp.sqrt(self.var + 1e-8)

            self.dbeta = cp.sum(outputGradient, axis=0) * 1/m
            self.dgamma = outputGradient * self.X_norm * 1/m

            dX_norm = outputGradient * self.gamma
            dvar = cp.sum(dX_norm * X_mu, axis=0) * - \
                0.5 * (self.var + 1e-8)**(-3/2)
            dmu = cp.sum(dX_norm * -var_inv, axis=0) + dvar * \
                1/self.n_X * cp.sum(-2. * X_mu, axis=0)
            dX = (dX_norm * var_inv) + (dmu / self.n_X) + \
                (dvar * 2/self.n_X * X_mu)

            dX = dX.reshape(self.xshape)

        return dX

    def update(self, lr, optim):
        if self.optimizer == None:
            if optim == "adam":
                self.optimizer = AdamOptim()
            if optim == "sgd":
                self.optimizer = SGD()
        self.dgamma = self.dgamma.sum(axis=0)
        self.gamma, self.beta = self.optimizer.optimize(
            self.t, lr, self.dgamma, self.dbeta, self.gamma, self.beta)
        self.t += 1

    def initializeLayer(self, x):
        if not self.initialized:
            if type(x) == np.ndarray:
                if len(x.shape) == 2:

                    self.n_X = x.shape[1]
                    self.gamma = np.ones((1, x.shape[0]))
                    self.beta = np.zeros((1, x.shape[0]))
                elif len(x.shape) == 4:

                    shape = x.shape[1]*x.shape[2]*x.shape[3]
                    self.n_X = x.shape[0]
                    self.gamma = np.ones((1, shape))
                    self.beta = np.zeros((1, shape))
            else:
                if len(x.shape) == 2:

                    self.n_X = x.shape[1]
                    self.gamma = cp.ones((1, x.shape[0]))
                    self.beta = cp.zeros((1, x.shape[0]))
                elif len(x.shape) == 4:

                    shape = x.shape[1]*x.shape[2]*x.shape[3]
                    self.n_X = x.shape[0]
                    self.gamma = cp.ones((1, shape))
                    self.beta = cp.zeros((1, shape))
            self.backshape = x.shape
            self.t = 1
            self.initialized = True

    def summary(self):
        summary = "Batch normalizer"
        return summary, False
