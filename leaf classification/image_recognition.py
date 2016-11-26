import numpy as np
import os, sys
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

__author__ = 'Ming Li'
# This application forms a submission from Ming in regards to leaf classification challenge on Kaggle community

path = 'leaf/images/'
pic_names = {i: path + str(i) + '.jpg' for i in range(1, 1585, 1)}


# exploring possible feature

pic = pic_names[99]




# Create the Keras model

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                        input_shape=(96, 96, 1), dim_ordering="tf"))

model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# layter 1

model.add(Convolution2D(32, 6, 6))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# layter 2

model.add(Convolution2D(32, 6, 6))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# layter 3

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(99))
model.add(Activation('sigmoid'))


model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])