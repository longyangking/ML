'''
Artificial Intelligence Model
'''
from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np

import keras 
import tensorflow as tf
import keras.backend as K

from keras.models import Model 
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, Add
from keras.optimizers import Adam
from keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

class NeuralNetwork:
    

class AI: