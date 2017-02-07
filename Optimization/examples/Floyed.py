import numpy as np

def Floyed(n,cost):
    A = np.zeros((n,n))
    P = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i,j] = cost[i,j]
            P[i,j] = 0

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if A[i,k] + A[k,j] < A[i,j]:
                    A[i,j] = A[i,k] + A[k,j]
                    P[i,j] = k

    return A,P

def path(P,i,j):
    k = P[i,j]
    if k is not 0:
        return np.array([path(P,i,k),k,path(P,k,j)]).flatten()
