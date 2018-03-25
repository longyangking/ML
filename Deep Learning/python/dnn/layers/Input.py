import numpy as np 

class Input:
    def __init__(self,shape=None):
        if shape is not None:
            self.shape = shape
        else:
            self.shape = (1,)

    def gettype(self):
        return 'Input'

    def getinfo(self):
        return self.shape