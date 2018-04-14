from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers
import keras.backend as K
import tensorflow as tf

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown


class MnistRes:
    def __init__(self,input_size, output_size, hidden_layers,learning_rate=1e-4,momentum=0.9,l2_const=1e-4,verbose=False):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.momentum = momentum

        self.model = None

        self.verbose = verbose

    def init(self):
        if self.verbose:
            print("Initiating model ...",end="")

        main_input = Input(shape = self.input_size, name = 'main_input')

        x = self._conv_block(main_input, self.hidden_layers[0]['nb_filter'], self.hidden_layers[0]['kernel_size'])
        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self._res_block(x, h['nb_filter'], h['kernel_size'])

        value = self._policy_value_block(x)

        self.model = Model(inputs=[main_input], outputs=[value])
        self.model.compile(loss={'value_head': 'categorical_crossentropy'},
			optimizer=SGD(lr=self.learning_rate, momentum=self.momentum)	
			)

        if self.verbose:
            print("Successful!")

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, verbose=0)
        return score
    
    def predict(self, X):
        return self.model.predict(X)

    def update(self,X_train, y_train , epochs=100, batch_size=128, validation_split=0.2):
        self.model.fit(X_train,y_train, epochs=epochs, verbose=self.verbose, validation_split = validation_split, batch_size = batch_size)
    
    def _policy_value_block(self,input_tensor):
        out = Conv2D(
            filters = 1,
            kernel_size = (1,1),
            data_format="channels_first",
            padding = 'same',
            use_bias=False,
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
		)(input_tensor)

        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        out = Flatten()(out)

        out = Dense(
			20,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.l2_const)
		)(out)

        out = LeakyReLU()(out)

        value = Dense(
			self.output_size, 
            use_bias=False,
            activation='softmax',
            kernel_regularizer=regularizers.l2(self.l2_const),
            name = 'value_head'
			)(out)

        return value

    def _conv_block(self, input_tensor, nb_filter, kernel_size=3):
        out = Conv2D(
            filters = nb_filter,
            kernel_size = kernel_size,
            data_format="channels_first",
            padding = 'same',
            use_bias=False,
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
		)(input_tensor)

        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)

        return out

    def _res_block(self, input_tensor, nb_filter, kernel_size=3):
        out = self._conv_block(input_tensor, nb_filter, kernel_size)

        out = Conv2D(
                filters = nb_filter,
                kernel_size = kernel_size,
                data_format="channels_first",
                padding = 'same',
                use_bias=False,
                activation='linear',
                kernel_regularizer = regularizers.l2(self.l2_const)
		    )(out)

        out = BatchNormalization(axis=1)(out)
        out = add([input_tensor, out])
        out = LeakyReLU()(out)

        return out

def data_augement(self, train_data):
    # 1. Add small noise
    # 2. Compress or amplify
    # 3. Deform
    # 4. Remove some pixels randomly 
    pass

if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_rows, img_cols = x_train.shape[1:]
    num_classes = 10

    input_size = (1,28,28)
    output_size = (10)
    hidden_layers = list()
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    model = MnistRes(input_size, output_size, hidden_layers, verbose=True)
    model.init()

    X_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    X_test = x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    model.update(X_train=X_train, y_train=y_train)

    score = model.evaluate(X_test, y_test)
    print('Test loss:', score)
