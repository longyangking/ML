import numpy as np 

class Model:
    def __init__(self,inputs,outputs,name=None):

    def compile(self,optimizer,loss,
                metrics=None, loss_weights=None,sample_weight_mode=None,
                weighted_metrics=None,target_tensors=None):


    def evaluate(self,x,y,batch_size=None,verbose=1,sample_weight=None,steps=None):


    def fit(self,x=None,y=None,batch_size=None,epochs=1,verbose=1,callbacks=None,
            validation_split=0.0,validation_data=None,shuffle=True,class_weight=None,
            sample_weight=None,initial_epoch=0,steps_per_epoch=None,validation_steps=None):

    def predict(self,x,batch_size=None,verbose=0,steps=None):

    def predict_on_batch(self,x):


    def test_on_batch(self,x,y,sample_weight=None):

    def train_on_batch(self,x,y,sample_weight=None,class_weight=None):

