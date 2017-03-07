"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from numpy import loadtxt, ones, zeros, where
from sklearn import preprocessing

from nn import NeuralNet

# Load Data
filename1 = 'data/digitsX.dat'
data = loadtxt(filename1, delimiter=',')
X_tobe = data[:, :]
min_max_scaler = preprocessing.MinMaxScaler()
X=X_tobe#min_max_scaler.fit_transform(X_tobe)
n, d = X.shape

filename2 = 'data/digitsY.dat'
data = loadtxt(filename2, delimiter=',')
#print data
y = data[:]

Xtrain = X#[0:4000,:]
ytrain = y#[0:4000]

Xtest=X[4501:5000,:]
ytest=y[4501:5000]
#print X[0:10,:]
# n,d=(X_tobe).shape
# for i in range(n):
    # X[i]=(X_tobe[i]-X_tobe[i].min())/X_tobe[i].min()
# train the boosted DT
hidden = np.array([25])
modelBoostedDT = NeuralNet(layers=hidden, epsilon=0.12, learningRate=2.18, numEpochs=750)
modelBoostedDT.fit(Xtrain,ytrain)


#print ytest[1:10]
# output predictions on the remaining data
ypred_BoostedDT = modelBoostedDT.predict(Xtrain)
#for i in range(len(ytest)):
#    print ytest[i],ypred_BoostedDT[i]
print accuracy_score(ytrain,ypred_BoostedDT)
modelBoostedDT.visualizeHiddenNodes("king.jpg")
#print np.unique(ypred_BoostedDT)
#print ypred_BoostedDT[0:999]
#print ypred_BoostedDT[999:1999]
#print ypred_BoostedDT[1999:2999]
#print ypred_BoostedDT[2999:3999]
