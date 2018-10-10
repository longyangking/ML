import numpy as np 
import utils

class NeuralNetwork:
    def __init__(self,layers,activation='tanh',initialweights=None):
        self.activation = utils.tanh
        self.activation_deriv = utils.tanh_deriv

        if activation == 'logistic':
            self.activation = utils.logistic
            self.activation_deriv = utils.logistic_deriv
        elif activation == 'relu':
            self.activation = utils.relu
            self.activation_deriv = utils.relu_deriv

        # Initiate Weight Tensor
        if initialweights is None:
            self.weights = list()
            for i in range(1,len(layers) - 1): # First and last layer for input and output, respectively
                self.weights.append(\
                    (2*np.random.random((layers[i-1] + 1, layers[i] + 1))-1)*0.25)    
            self.weights.append(\
                    (2*np.random.random((layers[i] + 1, layers[i+1]))-1)*0.25)
        else:
            self.weights = initialweights

    def fit(self,X,y,learning_rate=0.2,epochs=10000):
        '''
        X: Train set. Row means the sample number. Column means the feature
        y: Label set, only 1 label allowed currently
        '''
        X = np.atleast_2d(X) # Make sure that arrays with at least two dimensions
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,0:-1] = X    # Add bias term
        X = temp
        y = np.array(y) # Make sure that y is of Numpy format

        # Stochastic gradient descent
        for k in range(epochs):
            index = np.random.randint(X.shape[0])
            values = [X[index]]

            # To calculate the values of each layers
            for layer in range(len(self.weights)):
                #print '*',layer
                #print np.size(values[layer])
                #print np.size(self.weights[layer],0)
                values.append(self.activation(np.dot(values[layer],self.weights[layer])))
            error = y[index] - values[-1]

            # Back-propagation of error
            deltas = [error*self.activation_deriv(values[-1])]
            for layer in range(len(values)-2,0,-1):
                deltas.append(\
                    deltas[-1].dot(self.weights[layer].T)*self.activation_deriv(values[layer]))
            deltas.reverse()    # The first element of original set should belong to the last one. 

            # Update Weights
            for i in range(len(self.weights)):
                value = np.atleast_2d(values[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate*value.T.dot(delta)

    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x  # Add bias term
        pred = temp
        for layer in range(0,len(self.weights)):
            pred = self.activation(np.dot(pred,self.weights[layer]))
        return pred