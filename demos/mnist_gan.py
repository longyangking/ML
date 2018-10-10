import keras
import tensorflow as tf 

from keras.models import Model 
from keras.layers import Conv2D, Flatten, Input, MaxPooling2D, Dense, Activation
from keras.datasets import mnist
from keras import backend as K

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print('Channel status : {0}'.format(K.image_data_format()))



