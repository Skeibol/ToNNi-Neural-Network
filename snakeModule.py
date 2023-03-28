import numpy as np
import pickle
import math
from dataloader import Dataloader
import os
from utils import oneHot, appendData


class SnakeModule:
    def __init__(self) -> None:
        pass

    def loadData(split=True, printShapes=False, append=True):
        import os
        folder = "Data"
        file = "data.txt"
        path = os.path.join(folder, file)
        with open(path, "r") as a:
            lines = a.readlines()

        rawdata = []
        for i in lines:
            l = i.split(" ")
            # print(type(l[1]))
            rawdata.append(l[:10])

        data = np.array(rawdata).astype(np.float32)
        if append:
            data = appendData(data)
        data = data[data[:, -1].argsort()]

        data = data#[:35000]

        unique, counts = np.unique(data[:, -1], return_counts=True)

        print(dict(zip(unique, counts)))

        if split:
            X_train, Y_train, X_test, Y_test = splitDataset(data, printShapes)

        if split:
            return X_train.T, oneHot(Y_train), X_test.T, oneHot(Y_test)
        else:
            return data

    def writeData(modelName):
        labels = []
        data = getWeights(modelName)
        folder = "Data"
        for i in range(math.ceil(len(data)/2)):
            labels.append("W"+str(i+1))
            labels.append("b"+str(i+1))

        for i, layer in enumerate(data):
            file = f"{labels[i]}.txt"
            with open(os.path.join(folder, file), "w") as file:
                for j, row in enumerate(layer):
                    for i, member in enumerate(row):

                        if i < len(row)-1:
                            file.write(str(member)+",")
                        else:
                            file.write(str(member))
                    if j < len(layer)-1:
                        file.write("\n")


def getWeights(modelName):
    l = []

    model = Dataloader.load(modelName)
    for layer in model.network:
        if layer.getWeights() is not None:

            for num in range(2):
                if type(layer.getWeights()[num]) == np.ndarray:
                    l.append(layer.getWeights()[num])
                else:

                    l.append(layer.getWeights()[num].get())

    return l


def splitDataset(data, printShapes):
    np.random.shuffle(data)
    X = data[:, :-1].astype(np.float32)
    Y = data[:, -1].astype(np.int8)
    split = 0.999999
    X_train = X[0:int(X.shape[0]*split)]
    X_test = X[int(X.shape[0]*split):]
    Y_train = Y[0:int(X.shape[0]*split)]
    Y_test = Y[int(X.shape[0]*split):]
    return X_train, Y_train, X_test, Y_test
