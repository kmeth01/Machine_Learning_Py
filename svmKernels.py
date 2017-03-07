import numpy as np
import math

_polyDegree = 2
_gaussSigma = 10


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    #print ret.shape
    ret=(np.dot(X1,X2.T)+1)**2
    return ret



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n,d=X2.shape
    dist=np.sum((X1-X2[0,:])**2, axis=1)
    for i in range(1,n):
        temp=np.sum((X1-X2[i,:])**2, axis=1)
        dist=np.c_[dist,temp]
    return np.exp(-dist/2*(_gaussSigma)**2)
        



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1,d1=X1.shape
    n2,d2=X2.shape
    L2NormX1=(np.linalg.norm(X1,axis=1))
    L2NormX2=(np.linalg.norm(X2,axis=1))
    (L2NormX1).shape=(n1,1)
    (L2NormX2).shape=(n2,1)
    Div=L2NormX1.dot(L2NormX2.T)
    cosine=np.dot(X1,X2.T)/Div
    #print cosine.shape,"  ",(np.linalg.norm(X1,axis=1)).shape,"  ",(np.linalg.norm(X2,axis=1)).shape
    print cosine[0:10,0:10]
    return cosine

