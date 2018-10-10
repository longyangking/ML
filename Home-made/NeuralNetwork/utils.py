import numpy as np 

# Basic Activation Function

def tanh(x):
    '''
    Activation function: Tanh 
    '''
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    '''
    Activation function: Logistic
    '''
    return 1.0/(1.0 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x)*(1.0 - logistic(x))

def relu(x):
    '''
    Activation function: ReLU
    '''
    return x*(x>0)

def relu_deriv(x):
    # Lack of careful theoretical support around x = 0 in derivative of ReLU
    return 1.0*(x>0)