#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.layers import  Dropout, Conv2D, MaxPool2D
from keras.models import Sequential

def main():
    #load the model structure
    model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=(input_width, input_height, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1', input_shape=(900, 900, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
    model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
    model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
    model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))

    # Compared to the original VGG16, we skip the next 2 MaxPool layers,
    # and go ahead with dilated convolutional layers instead

    model.add(Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_1'))
    model.add(Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_2'))
    model.add(Conv2D(512, (3, 3), dilation_rate=(2, 2), activation='relu', name='conv5_3'))

    # Compared to the VGG16, we replace the FC layer with a convolution

    model.add(Conv2D(4096, (7, 7), dilation_rate=(4, 4), activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu', name='fc7'))
    model.add(Dropout(0.5))
    # Note: this layer has linear activations, not ReLU
    model.add(Conv2D(21, (1, 1), activation='linear', name='fc-final'))


    zeros = np.zeros((1,900,900,3),dtype=np.uint8)


    prob = model.predict(zeros,verbose=1)

    print("done1!")

    zeros2 = np.zeros((3,900,900,3),dtype=np.uint8)


    prob = model.predict(zeros2,verbose=1)

    print("done!")


if __name__ == "__main__":
    main()
