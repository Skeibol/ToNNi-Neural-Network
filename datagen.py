import numpy as np
import random
import cv2
from utils import zoom_at,standardize,translate,rotate_image
import matplotlib.pyplot as plt

class DataGenerator():
    def __init__(self,chance=0.5,setStandardVariation=True,augment=True,show=False):
        self.chance = 1 - chance
        self.std = setStandardVariation
        self.augment = augment
        self.show = show
       
    def augPicker(self):
        number = random.randint(0, 100) 
        
        if number < self.chance * 100:
            return None
        number = 100 - number
        CHOICES = 7
        split = (100 - (100*self.chance)) / CHOICES
        if number < split:
            dice = random.randint(0,3)
            
            if dice ==3:
                return ["blur","translate"]
            return ["blur"]
        elif number < split*2:
            dice = random.randint(0,4)
            if dice == 1:
                return ["erode","erode"]
            elif dice ==2:
                return ["erode","dilate"]
            elif dice ==3:
                return ["dilate","translate"]
            return ["zoomIn"]
        elif number < split*3:
            dice = random.randint(0,4)
            if dice == 1:
                return ["zoomOut","erode"]
            elif dice ==2:
                return ["zoomOut","rotate"]
            elif dice ==3:
                return ["zoomOut","translate"]
            return ["zoomOut"]
        elif number < split*4:
            dice = random.randint(0,4)
            if dice == 1:
                return ["erode","rotate"]
            elif dice ==2:
                return ["erode","zoomOut"]
            elif dice ==3:
                return ["dilate","translate"]
            return ["erode"]
            
        elif number < split*5:
            dice = random.randint(0,4)
            if dice == 1:
                return ["dilate","zoomIn"]
            elif dice ==2:
                return ["dilate","rotate"]
            elif dice ==3:
                return ["dilate","translate"]
            return ["dilate"]
        elif number < split*6:
            dice = random.randint(0,5)
            if dice == 1:
                return ["translate","zoomIn"]
            elif dice ==2:
                return ["translate","zoomOut"]
            elif dice ==3:
                return ["translate","erode"]
            elif dice ==4:
                return ["translate","dilate"]
            
            return ["translate"]
        elif number < split*6.5:
            dice = random.randint(0,6)
            if dice == 1:
                return ["translate","zoomIn"]
            elif dice ==2:
                return ["translate","zoomOut"]
            elif dice ==3:
                return ["translate","erode"]
            elif dice ==4:
                return ["translate","dilate"]
            return ["rotate"]

    def applyAug(self, X, augments):
        Xshape = X.shape
        
        for aug in augments:
            if aug == "flip":
                X = cv2.flip(X, 1)
            if aug == "zoomOut":
                X = zoom_at(X, 0.6)
            if aug == "zoomIn":
                X = zoom_at(X, 1.3)
            if aug == "erode":
                X = cv2.erode(X,(3,3))
            if aug == "translate":
                X = translate(X,5)
            if aug == "dilate":
                X = cv2.dilate(X,(3,3))
            if aug == "rotate":
                angle = random.randint(-65,65)
                X = rotate_image(X,angle)
            if aug == "blur":
                X = cv2.blur(X,(3,3))
            if len(X.shape) == 2:
                X = np.expand_dims(X,axis=2)

        assert Xshape == X.shape
        return X

    def generate(self, X_train, Y_train,X_test,Y_test):
        xShape = (1,X_train.shape[1],X_train.shape[2],X_train.shape[3])
        progress = 0
        newX = X_train.copy()
        newY = Y_train.copy()
        if self.show:
            fig = plt.figure(figsize=(8, 8))
            columns = 4
            rows = 5
            c = 1

        if self.augment:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], -1)
            for X, Y in zip(X_train, Y_train.T):
                progress+=1
                print(f"\rData generating: {(progress/X_train.shape[0])*100:.2f}%",end="")
                aug = self.augPicker()
                if aug == None:
                    continue
                
                X = self.applyAug(X, aug)
                if self.show and c<columns*rows + 1:
                    
                    
                    fig.add_subplot(rows, columns, c)
                    
                    if xShape[1] == 1:
                        plt.imshow(X,cmap="gray")
                    else:
                        
                        plt.imshow(X/255)
                        #working_img = cv2.cvtColor(X,cv2.COLOR_RGB2BGR )
                    plt.title(f"{np.argmax(Y)} , {aug}")
                    c+=1
                    #cv2.imshow(f"{np.argmax(Y)} , {aug}", X/255)
                    #cv2.waitKey()
                #
                    #cv2.destroyAllWindows()
                #print(X.shape)
                assert X.reshape(xShape).shape == xShape
                newX = np.append(newX,X.reshape(xShape),axis=0)
                newY = np.append(newY,np.expand_dims(Y,axis=1),axis=1)
        
        if self.show and self.augment:
            fig.tight_layout(pad=1.5)
            plt.show()
        if self.std:
            newX = self.stand(newX)
            X_test = self.stand(X_test)
            
        return newX,newY,X_test,Y_test
    
    def stand(self,X_train):
        return standardize(X_train)
