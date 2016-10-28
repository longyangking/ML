#!/usr/bin/python
import numpy as np
from numpy import cov,mean,transpose,matrix
from numpy.linalg import inv,det
from math import log,sqrt

def Bhattacharyya(p,q):
    p = np.matrix(p); q = np.matrix(q)
    sigmap = cov(p.T); sigmaq = cov(q.T)
    mup = mean(p,axis=0); muq = mean(q,axis=0)
    sigma = (sigmaq + sigmap)/2.0
    distance = (mup-muq)*inv(sigma)*transpose(mup-muq)/8.0 + log(det(sigma)/sqrt(det(sigmap)*det(sigmaq)))
    return distance

if __name__ == '__main__':
    print 'Self Test'
    print "Bhattacharyya Distance = {distance}".format(distance=Bhattacharyya(np.random.normal(1,0.1,size=[10,5]),np.random.normal(1.2,0.2,size=[10,5])))
