import numpy as np
import math


def loadData(split=True,printShapes=False):
    with open("Data/data.txt") as file:
        lines = file.readlines()
        

    rawdata = []
    for i in lines:
        l = i.split(" ")
        
        rawdata.append(l[:10])

    data = np.array(rawdata)
    

    if(split): X_train, Y_train, X_test, Y_test = splitDataset(data,printShapes)
    

    if split: 
        return X_train, Y_train, X_test, Y_test 
    else:
        return data

def splitDataset(data,printShapes):
    np.random.shuffle(data)
    X = data[:,:-1].astype(np.float32)
    Y = data[:,-1].astype(np.int8)
    split = 1
    X_train = X[0:int(X.shape[0]*split)]
    X_test = X[int(X.shape[0]*split):]
    Y_train = Y[0:int(X.shape[0]*split)]
    Y_test = Y[int(X.shape[0]*split):]
    

    if(printShapes):
        print("Y_train : ",Y_train.shape)
        print("Y_test : ",Y_test.shape)
        print("X_train : ",X_train.shape)
        print("X_test : ",X_test.shape)

    return X_train, Y_train, X_test, Y_test

def writeData(data):
    labels=[]
    for i in range(math.ceil(len(data))):
        labels.append("W"+str(i+1))
        labels.append("b"+str(i+1))
    
    c=0
    c1=1
    for layer in data:
        
        with open(f".\\Model\\{labels[c]}.txt", "w+") as file:
            for i,weight in enumerate(layer.W):
                for j,w in enumerate(weight):
                    
                    if j<layer.W.shape[1] - 1:
                        file.write(str(w)+",")
                    else:
                        file.write(str(w))
                
                if i<layer.W.shape[0] - 1:
                    file.write("\n")
                    
                
        with open(f".\\Model\\{labels[c1]}.txt", "w+") as file:
            for i,bias in enumerate(layer.b):
                for j,b in enumerate(bias):
                    
                    if j<layer.b.shape[1] - 1:
                        file.write(str(b)+",")
                    else:
                        file.write(str(b))

                if i<layer.b.shape[0] - 1:
                    file.write("\n")                    
                
        c1+=2
        c+=2



    


        