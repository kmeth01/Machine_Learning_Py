'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''

import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

class BoostedDT:

    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        self.numBoostingIters=numBoostingIters
        self.MaxTreeDepth=maxTreeDepth
        self.H_DT=[None]*self.numBoostingIters
        self.alpha=np.zeros(self.numBoostingIters)
        self.numlabels=None
        self.labels=None        
    

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d=X.shape
        self.labels=np.unique(y) 
        self.numlabels=self.labels.size
        sample_wt=np.ones(n)/n
        for i in range(self.numBoostingIters):
             self.H_DT[i]=tree.DecisionTreeClassifier(max_depth=self.MaxTreeDepth)
             self.H_DT[i].fit(X,y,sample_weight=sample_wt)
             y_pred=self.H_DT[i].predict(X)
             wt_error=((y_pred!=y)*sample_wt).sum()
             self.alpha[i]=0.5*(np.log((1-wt_error)/wt_error)+np.log(self.numlabels-1))
             for j in range(n):
                 if y_pred[j]==y[j] :
                    sample_wt[j]=sample_wt[j]*np.exp(-1*self.alpha[i])
                 else:
                    sample_wt[j]=sample_wt[j]*np.exp(self.alpha[i])
             sample_wt=sample_wt/sample_wt.sum()

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d=X.shape
        preds=np.zeros((n,self.numlabels))
        for i in range(self.numBoostingIters):
            y_pred=self.H_DT[i].predict(X)
            for j in range(self.numlabels):
                preds[:,j]=preds[:,j]+(y_pred==self.labels[j])*self.alpha[i]
        y=self.labels[np.argmax(preds,axis=1)]
        return y
        