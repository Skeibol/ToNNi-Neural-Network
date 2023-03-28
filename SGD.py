class SGD:
    def __init__(self) -> None:
        pass

    def optimize(self,t,lr,dW,dB,W,b):
        W = W - lr * dW
        b = b - lr * dB

        return W,b

        