from utils import oneHot,import_tensorflow
import numpy as np
import os
import pickle
import cv2

class Dataloader:
    def __init__(self):     
        pass

    def loadMNIST(batchsize=-1,valsize=-1,onehot=True,binary=False,gray=False,fashion=False):
        tf = import_tensorflow() 
        if not fashion:
            (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
        else:
            (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.fashion_mnist.load_data()

        batchsize = batchsize
        valsize = valsize
        onehot = onehot

        if binary:
            batchsize = 12000 # 18000 - 2 klase
            valsize = 1200
            arr1inds = Y_train.ravel().argsort()
            X_train = X_train[arr1inds,:,:]
            Y_train = Y_train[arr1inds]
            arr1inds = Y_test.ravel().argsort()
            X_test = X_test[arr1inds,:,:]
            Y_test = Y_test[arr1inds]
            Y_train = np.expand_dims(Y_train,axis=0)
            Y_test = np.expand_dims(Y_test,axis=0)
            if onehot:
                print("ONE HOT ON BINARY OUTPUT!!")

        X_train =  np.reshape(X_train,(-1,1,28,28)).astype(np.float32)
        X_train = X_train[:batchsize] 

        X_test =  np.reshape(X_test,(-1,1,28,28)).astype(np.float32)
        X_test = X_test[:valsize] 
        
        if onehot:
            Y_train = oneHot(Y_train[:batchsize].T)
            Y_test = oneHot(Y_test[:valsize].T)
        else:
            Y_train = Y_train[:,:batchsize]
            Y_test = Y_test[:,:valsize]

        if batchsize != -1:
            assert Y_train.shape[1] == batchsize
        if valsize != -1:
            assert Y_test.shape[1] == valsize  

        if not fashion:
            print("MNIST ")
            if binary: print("BINARY ")
            if onehot:
                print("ONEHOT ") 
            else: 
                print("NO ONEHOT ")
        return X_train, Y_train, X_test, Y_test

    def loadFashionMNIST(batchsize=-1,valsize=-1,onehot=True,binary=False,gray=False):
        print("FASHION MNIST ")
        if binary: print("BINARY ")
        if onehot:
            print("ONEHOT ") 
        else: 
            print("NO ONEHOT ")
        X_train, Y_train, X_test, Y_test = Dataloader.loadMNIST(batchsize,valsize,onehot,binary,gray,True)
        return X_train, Y_train, X_test, Y_test


    def loadCIFAR(batchsize=-1,valsize=-1,onehot=True,binary=False,gray=False):
        tf = import_tensorflow() 
        (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
        batchsize = batchsize
        valsize = valsize
        onehot = onehot

        if binary:
            batchsize = 10000 # 18000 - 2 klase
            valsize = 900
            arr1inds = Y_train.ravel().argsort()
            X_train = X_train[arr1inds,:,:,:]
            Y_train = Y_train[arr1inds]
            arr1inds = Y_test.ravel().argsort()
            X_test = X_test[arr1inds,:,:,:]
            Y_test = Y_test[arr1inds]
            
            #Y_train = np.expand_dims(Y_train,axis=0)
            #Y_test = np.expand_dims(Y_test,axis=0)
            if onehot:
                print("ONE HOT ON BINARY OUTPUT!!")
        if gray:
            print("GRAY CIFAR ")
            X_train = np.array([cv2.cvtColor(np.reshape(i,(32,32,3)),cv2.COLOR_BGR2GRAY) for i in X_train])
            X_test = np.array([cv2.cvtColor(np.reshape(i,(32,32,3)),cv2.COLOR_BGR2GRAY) for i in X_test])

            X_train =  np.reshape(X_train,(-1,1,32,32)).astype(np.float32)
            X_test =  np.reshape(X_test,(-1,1,32,32)).astype(np.float32)

        else:
            print("RGB CIFAR ")
            X_train =  np.reshape(X_train,(-1,3,32,32)).astype(np.float32)
            X_test =  np.reshape(X_test,(-1,3,32,32)).astype(np.float32)
        X_train = X_train[:batchsize] 
       

        X_test = X_test[:valsize] 
        
        if onehot:
            Y_train = oneHot(Y_train[:batchsize].T)
            Y_test = oneHot(Y_test[:valsize].T)
        else:
            Y_train = Y_train[:batchsize].T
            Y_test = Y_test[:valsize].T

        if batchsize != -1:
            assert Y_train.shape[1] == batchsize
        if valsize != -1:
            assert Y_test.shape[1] == valsize 

        if binary: print("BINARY ")
        if onehot:
            print("ONEHOT ") 
        else: 
            print("NO ONEHOT ")
        return X_train, Y_train, X_test, Y_test

    def loadPickledGTSRB(batchsize=39209,valsize=12630,onehot=True,binary=False,gray=False):
            import pickle
            with (open('./datasets/gtsrb_pickled.pkl', 'rb')) as f:
                X_train, Y_train, X_test, Y_test = pickle.load(f, encoding='bytes')
            ################ TEMPLATE START ################
            batchsize = batchsize
            valsize = valsize
            onehot = onehot 

            if binary: # glavna vs sporedna; 12vs13
                batchsize = 4000 # [18460,20460>,[20460,22460> #2000x2
                valsize = 1300 # 650x2
                # arr1inds = Y_train.ravel().argsort()    # train slozen po klasama [20,0,1,2,..,41,42]
                X_train = X_train[18460:22460,:,:,:]
                Y_train = Y_train[18460:22460]
                
                arr1inds = Y_test.ravel().argsort()
                X_test = X_test[arr1inds,:,:,:]
                X_test = X_test[5920:7220,:,:,:] # [5920,6570>,[6570,7220> #650x2
                Y_test = Y_test[arr1inds]
                Y_test = Y_test[5920:7220]  

                Y_train = np.where(Y_train == 12, 0, 1)
                Y_test = np.where(Y_test == 12, 0, 1)
                if onehot:
                    print("ONE HOT ON BINARY OUTPUT!!") 

            if gray:
                print("GRAY GTSRB ")
                X_train = np.array([cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in X_train])
                X_test = np.array([cv2.cvtColor(i,cv2.COLOR_BGR2GRAY) for i in X_test]) 

                X_train =  np.reshape(X_train,(-1,1,48,48)).astype(np.float32)
                X_test =  np.reshape(X_test,(-1,1,48,48)).astype(np.float32)    

            else:
                print("RGB GTSRB ")
                X_train =  np.reshape(X_train,(-1,3,48,48)).astype(np.float32)
                X_test =  np.reshape(X_test,(-1,3,48,48)).astype(np.float32)    

            X_train = X_train[:batchsize] 
            
            X_test = X_test[:valsize] 
                

            if onehot:
                Y_train = oneHot(Y_train[:batchsize].T)
                Y_test = oneHot(Y_test[:valsize].T)
            else:
                Y_train = Y_train[:batchsize].T
                Y_test = Y_test[:valsize].T 

            assert Y_train.shape[1] == batchsize
            assert Y_test.shape[1] == valsize   

            if binary:
                print("BINARY ")
            if onehot:
                print("ONEHOT ") 
            else: 
                print("NO ONEHOT ")
                
            return X_train, Y_train, X_test, Y_test
        
    def save(X_train,Y_train,X_test,Y_test,name="aug"):
        obj = (X_train,Y_train,X_test,Y_test)
            
        name = f"dataset_{name}.pkl"
        folder = "Model"
        if not os.path.exists(folder):
            os.makedirs(folder)

        path = os.path.join(folder,name)
        with open(path, 'wb') as outp:
            pickle.dump(obj , outp, pickle.HIGHEST_PROTOCOL)
        name = " ".join(name[:-4].split("_"))
        print(f"{name} saved")
            
    def load(name):
        name = f"{name}.pkl"
        folder = "Model"
        path = os.path.join(folder,name)
        with open(path, 'rb') as outp:
            obj = pickle.load(outp)
        name = " ".join(name[:-4].split("_"))
        
        print(f"{name} loaded")
        
        return obj
        
        
        
        
        
        
        
