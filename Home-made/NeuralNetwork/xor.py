import sys
sys.path.append("..")

import numpy as np 
import gandalf.neuralnetwork as neuralnetwork

if __name__ == '__main__':
    print '*'*10,'XOR','*'*10
    nn = neuralnetwork.NeuralNetwork([2,2,1],'tanh')
    x = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([1,0,0,1])
    nn.fit(x,y,0.1,10000)
    for i in [[0,0],[0,1],[1,0],[1,1]]:
        print (i,nn.predict(i))

    print '\n','*'*10,'Arbitary','*'*10
    nn = neuralnetwork.NeuralNetwork([2,2,1],'tanh')
    X = np.array([[0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    y = np.array([0,0,1,1])
    nn.fit(x,y,0.1,10000)

    for i in [[0,0],[0,1],[1,0],[1,1]]:
        print (i,nn.predict(i))