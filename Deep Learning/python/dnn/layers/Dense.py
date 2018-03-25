import numpy as np 

class Dense:
    #def __init__(self,units,activation=None,use_bias=True,kernel_initializer='glorot_uniform',
    #            bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,
    #            activity_regularizer=None,kernel_constraint=None,bias_constraint=None):
    def __init__(self,units,activation=None,init='random'):
        self.units = units
        if activation is not None:
            self.activation = activation

        self.init = init

    def gettype(self):
        return 'Dense'

    def getinfo(self):
        return (self.unit, self.activation, self.init)

    def __call__(self,inputs):
        pass
