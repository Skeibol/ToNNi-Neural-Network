import numpy as np
import cv2
import random


def oneHot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y.astype(np.int8)


def standardize(X_train):
    X_mean = np.mean(X_train)  # 120.7
    X_centered = X_train - X_mean  # mean mu je sad 3e-14
    X_var_std2 = np.var(X_centered)  # 4115.2  ## np.mean(X_centered ** 2)

    # var oko 1, mean oko 0: good
    X_norm_stand = X_centered / np.sqrt(X_var_std2 + 1e-5)
    # X_train = np.reshape(X_norm_stand,(-1,3,32,32)) # flip channels

    return X_norm_stand

# may put this function in another utility file


def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf


def zoom_at(img, zoom=1, angle=0, coord=None):

    cy, cx = [i/2 for i in img.shape[:-1]] if coord is None else coord[::-1]

    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, zoom)
    result = cv2.warpAffine(
        img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def translate(img, strength=3):
    h, w = img.shape[:2]

    quarter_height, quarter_width = h // strength, w // strength

    up = random.randint(-quarter_height, quarter_height)
    side = random.randint(-quarter_width, quarter_width)
    up = up * 1.2
    side = side * 1.2
    T = np.float32([[1, 0, up], [0, 1, side]])
    img_translation = cv2.warpAffine(img, T, (w, h))

    #cv2.imshow("Originalimage", img)
    #
    #cv2.imshow('Translation', img_translation)
    # cv2.waitKey()
  #
    # cv2.destroyAllWindows()

    return img_translation


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    #cv2.imshow('Translation', img_translation)
    # cv2.waitKey()
    #
    # cv2.destroyAllWindows()
    return result


def appendData(data):

    print(data.shape, "old")
    start = False
    TIMES = 4
    cash = 0
    newdata = np.zeros((data.shape[0], (data.shape[1]*(TIMES+1))-TIMES))
    pad = np.zeros((1, data.shape[1]-1))
    padme = True
    third = True
    fourth = True
    start = True
    for idx, sample in enumerate(data):
        if padme == False:
            if second == True:
                newdata[idx] = np.append(np.append(pad.repeat(3), data[idx-1, :-1]), sample)
                second = False
            elif third == True:
                newdata[idx] = np.append(np.append(np.append(pad.repeat(2),data[idx-2, :-1]), data[idx-1, :-1]), sample)
                third = False
            elif fourth == True:
                newdata[idx] = np.append(np.append(np.append(np.append(pad,data[idx-3,:-1]),data[idx-2, :-1]), data[idx-1, :-1]), sample)
                fourth = False
            else:
                newdata[idx] = np.append(np.append(np.append(np.append(data[idx-4,:-1],data[idx-3,:-1]),data[idx-2, :-1]), data[idx-1, :-1]), sample)
            
        elif padme == True:
            newdata[idx] = np.append(pad.repeat(4), sample)
            padme = False
            second = True
            third = True
            fourth = True
            

        
        if sample[8] <= 0.11 and start == False:
            start = True

            padme = True
            # print(cash)
            # print(sample[8])
        elif not (sample[8] <= 0.11) and start == True:
            start = False

        cash = sample[8]
    print(newdata.shape, "new")
    return newdata
