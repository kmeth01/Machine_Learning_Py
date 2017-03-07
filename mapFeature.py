import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    n=len(x1)
    expfeat=np.c_[x1,x2]
    main = expfeat.copy()
    for i in range(2,7):
        d=i+1
        for j in range(d):
            expfeat=np.c_[expfeat,np.multiply(x1**(i-j),x2**(j))]
    #print expfeat.shape       
    return expfeat


