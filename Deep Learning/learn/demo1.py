from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Merge
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

def res_block(input_tensor, nb_filters, kernel_size=3):
    out = Conv2D(nb_filters[0], 1, 1)(out)

    return out


if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train)