
import numpy as np

from src import Layer,dataHandler



class Model:
    def __init__(self, lr, epochs,output_size,batch_size=64):
        self.lr = lr
        self.epochs = epochs
        self.output_size = output_size
        self.batch_size = batch_size 
        
    def layer_init(self):

        self.Layers=[
                    Layer.Input(None),


                    Layer.Dense(self.input_size, 20,    activation="relu"),
                    Layer.Dense(20,              10,                 activation="relu"),
                    Layer.Dense(10,              10,                 activation="relu"),
                    Layer.Dense(10,self.output_size,   activation="softmax"),


                    Layer.Output(None)
                    ]
    
        
     
    def forward(self):
        for i,layer in enumerate(self.Layers):
            if i!=0 and i!=len(self.Layers) -1:
                layer.forward()

    def backward(self):
        for i,layer in enumerate(reversed(self.Layers)):
            if i!=0 and i!=len(self.Layers) -1:
                layer.backward()

    def update(self,lr):
        for i,layer in enumerate(self.Layers):
            if i!=0 and i!=len(self.Layers) -1:
                layer.update(lr)
        
    def flush(self):
        for i,layer in enumerate(self.Layers):
            if i!=0 and i!=len(self.Layers) -1:
                layer.flush()

    def get_predictions(self): 
        return np.argmax(self.Layers[len(self.Layers) -2].output, 0) 

    def get_accuracy(self,predictions, Y): 
        return np.sum(predictions == Y) / Y.size 

    def fit(self, X, Y,diagnostics=True,plot=False,save_best_model=True):  
        self.input_size = X.shape[0] 
        prevAccuracy = 0.0
        batches_X = np.array_split(X,self.batch_size,axis=1)
        batches_Y = np.array_split(Y,self.batch_size,axis=0)
        self.layer_init()
        for epoch in range(self.epochs):
            for i,batch in enumerate(batches_X):
                self.X = batch
                self.Y = batches_Y[i]
                self.Layers[0] = Layer.Input(self.X)
                self.Layers[-1] = Layer.Output(self.Y)
                Layer.compile(self.Layers,self.Y)


                self.forward()
                self.backward()
                self.update(self.lr)
                #self.flush()

                predictions = self.get_predictions()
                accuracy = self.get_accuracy(predictions, self.Y)

                model_paramaters = self.Layers[1:-1]
                if accuracy>0.80 and accuracy>prevAccuracy and save_best_model:
                    Wb = self.Layers[1:-1]
                    dataHandler.writeData(Wb)
                    prevAccuracy=accuracy

                ################################ DEBUGGING ###########################
                if diagnostics:
                    #if i%8 == 0:
                    #    print("Batch: ",i*len(self.Y))
                    if epoch % 50 == 0 and i==0: 
                        
                        print("\rEpoch: ", epoch)
                        print(np.asarray(((np.unique(predictions, return_counts=True)))).T)
                        print(np.asarray(((np.unique(self.Y, return_counts=True)))).T)
                        
                        print("Accuracy: ", accuracy)
                        print("Max Accuracy: ",prevAccuracy)
                        print("------------------------\n")
                ################################ DEBUGGING ###########################
            
            
        print("Max acc: ", prevAccuracy)
        
        
        return model_paramaters
    def evaluate(self,X,Y, W1, b1, W2, b2):

        _,_,_, A2 = self.forward(X,W1, b1, W2, b2)
        predictions = self.get_predictions(A2)
        return self.get_accuracy(predictions, Y)

    def predict(self,X,W1,b1,W2,b2):
        Z1 = W1.dot(X) + b1 
        A1 = self.ReLU(Z1) 
        Z2 = W2.dot(A1) + b2 
        A2 = self.softmax(Z2) 
        
        return Z2

    def load(path): #rework
        data = []
        with open(path) as file:
            lines = file.readlines()
            
        for i in lines:
            
            i = i.replace("\n","")
            
            data.append(i)
        data = np.array((data)).reshape(len(data),-1)
        return data

