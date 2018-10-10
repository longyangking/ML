import sys
sys.path.append("..")

import numpy as np 
import matplotlib.pyplot as plt
import gandalf.neuralnetwork as neuralnetwork

def sampledata(n):
    X = np.zeros((2*n,2))
    y = np.zeros(2*n)
    X[:n,0] = np.random.normal(loc=1.0, scale=0.5, size=[n])
    X[:n,1] = np.random.normal(loc=-1.0, scale=0.5, size=[n])

    X[n:,0] = np.random.normal(loc=-1.0, scale=0.5, size=[n])
    X[n:,1] = np.random.normal(loc=1.0, scale=0.5, size=[n])
    y[n:] = np.ones(n)
    return X,y

if __name__ == '__main__':
    N = 200
    X,y = sampledata(N)
    
    nn = neuralnetwork.NeuralNetwork([2,4,6,1],'tanh')    # Need to debug, i.e, when to set [2,5,5,1]
    nn.fit(X,y,0.1,10000)

    N0 = 100
    data = np.zeros((N0,N0))
    Xs = np.linspace(-5,5,N0)
    Ys = np.linspace(-5,5,N0)
    for i in range(N0):
        for j in range(N0):
            data[i,j] = nn.predict([Xs[i],Ys[j]])
    
    [Xs,Ys] = np.meshgrid(Xs,Ys)
    plt.pcolormesh(Xs,Ys,data)
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.show()