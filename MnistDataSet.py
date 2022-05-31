import numpy as np

import keras
print("activate")
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

def get_image(img_num):
    image = X_train[img_num]
    image_list = []
    for i, x in enumerate(image):
        for y in x:
            image_list.append(y)
    for i in range(len(image_list)):
        image_list[i] /= 255

    return image_list, y_train[img_num]
