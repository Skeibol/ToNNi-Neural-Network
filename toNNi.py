import numpy as np
import pickle 
import os
import copy
import matplotlib.pyplot as plt
from datetime import datetime

class Model:
    def __init__(self, network,loss,optimizer,lr, epochs,batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.lossfunction = loss
        self.optimizer = optimizer
        self.batch_size = batch_size 
        self.network = network
    
    def forward(self,input,training=True):
        for layer in self.network:    
            input = layer.forward(input,training)

    def backward(self,output):
        for layer in reversed(self.network):
            output = layer.backward(output)

    def update(self,lr,optimizer):
        for layer in self.network:
            layer.update(lr,optimizer)

    def fit(self, X, Y, X_test,Y_test,diagnostics=True,shuffle=True,diagnosticsPerEpoch=1,name="mnist",validate=True): 
        SSSSSNAKEMODE = False
        accuracy_max = 0.0
        val_loss_max = 99999999999
        val_accuracy_max = 0.0
        loss_pred_max = 99999999
        accuracy_prev = 0
        self.trainLosses, self.validLosses = [], []
        accuracy = 0
        if len(X.shape) == 2:
            self.xshape = X.shape[1]
        else:
            self.xshape = X.shape[0]
        print(self.xshape)
        starttime = datetime.now()
        self.shuffle_check = shuffle
        self.X = X
        self.Y = Y
        ################################ START TRAINING EPOCH #############################
        for epoch in range(self.epochs):
            self.shuffle()
            oldtime = datetime.now()
            progress=0
            loss = 0
        ################################ START BATCHING ############################
            for i in range(0,self.xshape,self.batch_size):
                if i+self.batch_size>self.xshape-self.batch_size:
                        break
                
                if len(X.shape) == 2:
                    Xbatch = self.X_shuffled[:,i:i+self.batch_size]
                    Ybatch = self.Y_shuffled[:,i:i+self.batch_size]
                else:
                    Xbatch = self.X_shuffled[i:i+self.batch_size]
                    Ybatch = self.Y_shuffled[:,i:i+self.batch_size]
           
                ###

                self.forward(Xbatch,True) 
                predictions = self.network[-1].output
                grad = self.lossfunction.calculateGradient(predictions,Ybatch)
                self.backward(grad) 
                self.update(self.lr,self.optimizer)

                ####
                if diagnostics:              
                    predictions = self.get_predictions()
                    progress+=self.batch_size
                    accuracy +=  self.get_accuracy(predictions,Ybatch)  
                    loss +=np.sum(self.lossfunction.loss(predictions,Ybatch))  
                    self.update_progress(progress/self.xshape,accuracy/progress,loss/progress,(epoch+1)/diagnosticsPerEpoch)
                    #self.checkNan()
                else:
                    progress+=self.batch_size
                    print(f"\rProgress: {(progress/self.xshape)*100}%",end="")

            accuracy = accuracy/self.xshape
            #if accuracy>accuracy_max:
            #    self.save(name)
            #    accuracy_max = accuracy
        ################################ END TRAINING EPOCH #############################
    

        ################################ VALIDATION SET ###########################
            
                
            if len(self.X.shape) == 2:
                if validate:
                    loss_pred = 0
                    test_accuracy = 0
                    for idx in range(0,Y_test.shape[1],self.batch_size):
                        if idx+self.batch_size>Y_test.shape[1]:
                            break
                        self.forward(X_test[:,idx:idx+self.batch_size],False)
                        test_pred = self.get_predictions()
                        loss_pred += np.sum(self.lossfunction.loss(test_pred,Y_test[:,idx:idx+self.batch_size])) 
                        test_accuracy += self.get_accuracy(test_pred,Y_test[:,idx:idx+self.batch_size])
                    self.validLosses.append(loss_pred)
                    self.trainLosses.append(loss)
                    
                else:
                    self.forward(Xbatch,False)  
                    self.trainLosses.append(loss)   
                
            else:
                if validate:
                    loss_pred = 0
                    test_accuracy = 0
                    for idx in range(0,Y_test.shape[1],self.batch_size):
                        if idx+self.batch_size>Y_test.shape[1]:
                            break
                        self.forward(X_test[idx:idx+self.batch_size],False)
                        test_pred = self.get_predictions()
                        loss_pred += np.sum(self.lossfunction.loss(test_pred,Y_test[:,idx:idx+self.batch_size])) 
                        test_accuracy += self.get_accuracy(test_pred,Y_test[:,idx:idx+self.batch_size])
                    val_accuracy_max = test_accuracy
                    self.validLosses.append(loss_pred)
                    self.trainLosses.append(loss)
                else:
                    self.forward(Xbatch,False)
                    self.trainLosses.append(loss)
            if loss_pred<loss_pred_max:
                        if SSSSSNAKEMODE:
                            self.save(name)
                            self.snakeSave(name)
                            print("Saved weights")
                        else:
                            self.save(name)
                            print("Saved network")
                            
                        loss_pred_max = loss_pred
                    
            if loss_pred<val_loss_max and validate:
                
                if SSSSSNAKEMODE:
                    self.snakeSave(name)
                    print("Saved weights")
                else:
                    self.save(name)
                    print("Saved model")
                val_loss_max = loss_pred
                
                    
        ################################ VALIDATION SET END ###########################
   


        ################################ DEBUGGING ####################################
               
            if epoch % diagnosticsPerEpoch == 0:  
                print("------------------------")
                if not diagnostics:
                    print("\n------------------------")
                    print("DIAGNOSTICS OFF!!!!!")  
                    self.predict(self.X)[0]
                    predictions = self.get_predictions()
                    accuracy = self.get_accuracy(predictions,self.Y) / predictions.shape[0]
                print(f"\rEpoch: {epoch+1}")
                print(f"Accuracy: {accuracy*100:.2f}%")
                print(f"Loss: {loss/self.X.shape[0]}")
                if validate:
                    print(f"Validation accuracy: {test_accuracy/Y_test.shape[1]*100:.2f}%")
                    print(f"Validation loss: {loss_pred/Y_test.shape[1]}")
                    print(f"Validation accuracy max: {val_accuracy_max/Y_test.shape[1]*100:.2f}")
                print(". . . . . . . . . . . . . .")
                print(f"Input complexity: {self.X.size}")
                print(f"Network saturation: {self.getParams()/self.X.size:2f}")
                print(f"Training time: {datetime.now() - starttime}")
                print(f"Epoch time: {datetime.now() - oldtime}")
                print(f"Shuffle: {shuffle}")
                print(f"Learning rate: {self.lr}, Batch size: {self.batch_size}, Dataset:{name}")
                print(f"Max Accuracy: {accuracy_max*100:.2f}%")
                print(f"Prev accuracy: {accuracy_prev*100:.2f}%")
                if SSSSSNAKEMODE:
                    self.snakeSave(name)
                    print("Saved weights")
                print("------------------------\n")
                accuracy_prev = accuracy
            accuracy = 0
        ################################ DEBUGGING ###################################

    def predict(self,sample,show=False):
        if(len(sample) == 1 ):
            sample = np.expand_dims(sample,axis=0)
        self.forward(sample,False),
        if show:
            self.show()
        if self.network[-1].output.shape[1] == 1:
            idx = np.argmax(self.network[-1].output)
            return self.network[-1].output,idx
        else:
            idx = np.argmax(self.network[-1].output,axis=0)
            return self.network[-1].output,idx[0]

    def summary(self):
        c = 0
        print(f"Learning rate: {self.lr} Batch size: {self.batch_size}")
        for layer in self.network:
            if layer.summary() is not None:
                if layer.summary()[1] == True:
                    c+=1
                    print("")
                    print("--------------------"*4)
                    print("\n")
                    print(f"Layer {c}:",layer.summary()[0])   
                    init = " ".join(layer.initialization)
                    print(str(self.optimizer),"optimizer,", f"{init} initialization")
                else:
                    print(layer.summary()[0])
        print("\n")

    def loss(self):
        return self.trainLosses,self.validLosses
               


    def update_progress(self,progress,accuracy,loss,epoch):
        WHITE = '\033[0m'
        CYAN = '\033[30m'
        message = '\r[{0}{1}] {2:.2f}% ------ | acc: {3:.1f}% | loss: {4:.4f} | lr: {5} epoch {6:.2f} batchsize: {7} | '.format\
            (
                 '#'*int(2*(progress*10)),
                 " "*int(20-(progress*10)*2),
                 progress*100,
                 int(accuracy*100),
                 loss,
                 self.lr,
                 epoch,
                 self.batch_size
                 )
        print(message,end="")  

    def show(self,abs=False):
        height = 10
        f, axarr = plt.subplots(height,len([i.show().shape for i in self.network if i.show() is not None]) + 1,squeeze=True) 
        f.set_figheight(15)
        f.set_figwidth(15)
        c=0
        for layer in self.network:
            
            if layer.show() is not None:
                for i in range(layer.output.shape[1]):
                    idx = np.random.randint(0,layer.output.shape[1])
                    if not abs:
                        axarr[i,c].imshow(layer.show()[idx],cmap="gray")
                    else:
                        axarr[i,c].imshow(np.abs(layer.show()[idx]),cmap="gray")
                    if i==9: 
                        break
                c+=1

  
        plt.show()  
    
    def getParams(self):
        params = 0
        for layer in self.network:
            if layer.getParams() is not None:
                params += layer.getParams()
        return params

    def get_predictions(self): 
        if type(self.network[-1].output) is np.ndarray:
            return self.network[-1].output
        return self.network[-1].output.get() 

    def get_accuracy(self,predictions,Y): 
        if self.lossfunction.id == "BCE":
            return np.sum(np.round(predictions) == Y)

        elif self.lossfunction.id == "CCE":
            return np.sum(np.argmax(predictions,axis=0) == np.argmax(Y,axis=0))
        
        elif self.lossfunction.id == "MSE": # acc beskorisna za regresiju
                    return 0
    def snakeSave(self,name):
        from snakeModule import SnakeModule
        SnakeModule.writeData(f"model_{name}")
    def save(self,name):
        name = f"model_{name}.pkl"
        folder = "Model"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path = os.path.join(folder,name)
        with open(path, 'wb') as outp:
            pickle.dump(self , outp, pickle.HIGHEST_PROTOCOL)

        print(f"{name} saved")

    def checkNan(self):
        if np.isnan(np.sum(self.network[-2].output)):
            print("\n-----------------------------")
            print("NaN detected, resetting layers")
            print("-----------------------------\n")
            self.network = copy.deepcopy(self.cached_network)
            self.reinitLayers()
            self.shuffle()
        
    def shuffle(self):
        if len(self.X.shape) == 4:
            if self.shuffle_check:
                indices = np.arange(self.X.shape[0])
                np.random.shuffle(indices)
                self.X_shuffled = self.X[indices]
                self.Y_shuffled = self.Y[:,indices]
            else:
                self.X_shuffled = self.X
                self.Y_shuffled = self.Y
        else:
            if self.shuffle_check:
                indices = np.arange(self.X.shape[1])
                np.random.shuffle(indices)
                self.X_shuffled = self.X[:,indices]
                self.Y_shuffled = self.Y[:,indices]
            else:
                self.X_shuffled = self.X
                self.Y_shuffled = self.Y

    def sumGradients(self): #Meme function
        gradientSum = 0
        mean = 0
        standard = 0
        food = 0
        div = 0
        div1 = 0
        RANDOMVALUEICHOSE = 10e8
        for layer in self.network:
            if layer.getGradients() is not None:
                grad = layer.getGradients()
                food += grad[1]
                mean += grad[2]
                standard += grad[3]
                gradientSum += np.sum(grad[0])
                div += grad[0].size
                div1 += 1
            
        return gradientSum / div / RANDOMVALUEICHOSE , food / div1 , mean / div1 , standard / div1

       



