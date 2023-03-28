from toNNi import Model
from dense import Dense
from flatten import Flatten
from activations import ReLU,Softmax,Sigmoid,Tanh,LeakyReLU
from dataloader import Dataloader
from maxpool import MaxPooling2D
from batchNormalizer import Batchnorm
from convolutional import Convolutional2D
from binarycrossentropy import BCE
from categoricalcrossentropy import CCE
from dropout import Dropout
from secret import gpu
from datagen import DataGenerator
from snakeModule import SnakeModule





#X_train, Y_train, X_test, Y_test = SnakeModule.loadData(printShapes=True)
#SnakeModule.writeData("model_snek")
#X_train, Y_train, X_test, Y_test = Dataloader.loadMNIST(50000,10000)

datagen = DataGenerator(
      chance = 0.35,
      setStandardVariation=True,
      augment=True,
      show=True
)
#model = Dataloader.load("model_snek")
#model.predict(X_train[1],False)

#X_train, Y_train, X_test , Y_test = Dataloader.loadCIFAR(50000,10000,True,False,False)
#X_train,Y_train,X_test , Y_test = datagen.generate(X_train,Y_train,X_test,Y_test)
#
<<<<<<< HEAD
#Datč.ćlk.k-ć-jłŁaloader.save(X_train,Y_train,X_test , Y_test,"cifarRGBaug")
=======
# Dataloader.save(X_train,Y_train,X_test , Y_test,"mnistAug50k")
#Dataloader.save(X_train,Y_train,X_test , Y_test,"cifarRGBaug")
>>>>>>> 991f00957ff0dc19e143c920d42c5514b4add15e
X_train,Y_train,X_test , Y_test = Dataloader.load("dataset_mnistAug50k")


initialization = "he_standard"

network = [
<<<<<<< HEAD
      #atchnorm(),
      #onvolutional2D(3,64),
      #axPooling2D(2,2),
=======
      #Batchnorm(),
      #Convolutional2D(3,128),
      #MaxPooling2D(2,2),
      #ReLU(),
      Batchnorm(),
      Convolutional2D(3,256),
      MaxPooling2D(2,2),
      ReLU(),
>>>>>>> 991f00957ff0dc19e143c920d42c5514b4add15e
      gpu(),
      ReLU(),
      Flatten(),      
      Batchnorm(),
<<<<<<< HEAD
      Dense(256,initialization=initialization),
      Batchnorm(),
      ReLU(),
      Dense(128,initialization=initialization),
      Batchnorm(),
      ReLU(),
=======
      
      Dropout(0.9),
      Dense(2048,initialization=initialization),
      Batchnorm(),
      ReLU(),
      Dropout(0.6),
      Dense(1024,initialization=initialization),
      Batchnorm(),
      ReLU(),
      Dropout(0.7),
      Dense(1024,initialization=initialization),
      Batchnorm(),
      ReLU(),    
>>>>>>> 991f00957ff0dc19e143c920d42c5514b4add15e
      Dense(10,initialization=initialization),
      Softmax(),

]

loss = CCE()
optimizer = "adam"
  
model = Model(
      network=network,
      loss=loss,
      optimizer=optimizer,
      lr=0.0001,
      epochs=2000050,
<<<<<<< HEAD
      batch_size=64
      )
#model = Dataloader.load("model_MNIST_26:20:26")
#model.batch_size = 64
=======
      batch_size=64 
      )
#model = Dataloader.load("model_snek")

>>>>>>> 991f00957ff0dc19e143c920d42c5514b4add15e
#model.lr = 0.005
#model.lr = 5e-05
model.fit(
      X_train,
      Y_train,
      X_test,
      Y_test,
      diagnostics=True,
      shuffle=True,
      name="snek",
      validate=True
      )





