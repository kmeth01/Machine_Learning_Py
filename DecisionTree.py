"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
from sklearn.ensemble import AdaBoostClassifier
from boostedDT import BoostedDT
from sklearn.svm import SVC

# load the data set
# load the data
filename = 'C:\Users\karth\Desktop\CIS_519\hw3_skeleton\hw3_skeleton\data\challengeTrainLabeled.dat'
filename_T='C:\Users\karth\Desktop\CIS_519\hw3_skeleton\hw3_skeleton\data\challengeTestUnlabeled.dat'
allData = np.loadtxt(filename, delimiter=',')
allData_T = np.loadtxt(filename_T, delimiter=',')

X = allData[:,:-1]
y = allData[:,-1]
#print allData_T[0]
X_T = allData_T[:,:]
n,d = X.shape
nTrain = 0.7*n 
# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]
# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

# train the decision tree
modelDT = DecisionTreeClassifier()
modelDT.fit(Xtrain,ytrain)

#print ypred_DT

# train the boosted DT
modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=3)
model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),n_estimators=100,random_state=13)
kfold=cross_validation.KFold(n=n,n_folds=2,random_state=13)
results=cross_validation.cross_val_score(model,X,y,cv=kfold)
modelBoostedDT.fit(Xtrain,ytrain)
model.fit(Xtrain,ytrain)
clf=SVC()
clf.fit(Xtrain,ytrain)
y_pred_rbf1=clf.predict(Xtest)
scores=list()
scores_rbf=list()

k_model=BoostedDT(numBoostingIters=140,maxTreeDepth=5)
k_model.fit(X,y)
y_pred_k=k_model.predict(X_T)

np.savetxt("predictions-BoostedDT.dat", np.array([y_pred_k]), delimiter=",", fmt="%d")

clf=SVC(C=10,gamma=0.1)
clf.fit(X,y)
y_pred_rbf=clf.predict(X_T)

np.savetxt("predictions-BestClassifier.dat", np.array([y_pred_rbf]), delimiter=",", fmt="%d")

'''
kf=cross_validation.KFold(n=n,n_folds=10)
print "Max_Depth ","Num_Iterations ", "Cross Validation Accuracy " 
for numiters in range(10,150,10):
    for maxdepth in range(2,6,1):
        scores_BDT=list()
        for train_index,test_index in kf:
            X_train_k,X_test_k=X[train_index],X[test_index]
            y_train_k,y_test_k=y[train_index],y[test_index]
            k_model=BoostedDT(numBoostingIters=numiters,maxTreeDepth=maxdepth)
            k_model.fit(X_train_k,y_train_k)
            y_pred_k=k_model.predict(X_test_k)
            scores_BDT.append(accuracy_score(y_test_k,y_pred_k))
        print  maxdepth,",",numiters , "," , np.mean(scores_BDT)

kf=cross_validation.KFold(n=n,n_folds=10)
Kernels=['rbf','poly','sigmoid']
degree=[3,4,5,6]
C_2d_range = [1e-2, 1e-1,1, 1e1,1e2]
gamma_2d_range = [1e-1, 1, 1e1]
print "SVM RBD KERNEL"
print "C_Value ","Gamma ", "Cross Validation Accuracy " 
for C_in in C_2d_range:
    for gamma in gamma_2d_range:
        scores_BDT=list()
        for train_index,test_index in kf:
            X_train_k,X_test_k=X[train_index],X[test_index]
            y_train_k,y_test_k=y[train_index],y[test_index]
            clf=SVC(C=C_in,gamma=gamma)
            clf.fit(X_train_k,y_train_k)
            y_pred_rbf=clf.predict(X_test_k)
            scores_BDT.append(accuracy_score(y_test_k,y_pred_rbf))
        print  C_in,",",gamma , "," , np.mean(scores_BDT)
print "SVM poly KERNEL"
print "C_Value ","degree", "Cross Validation Accuracy " 
for C in C_2d_range:
    for deg in [2,3,4,5,6]:
        scores_BDT=list()
        for train_index,test_index in kf:
            X_train_k,X_test_k=X[train_index],X[test_index]
            y_train_k,y_test_k=y[train_index],y[test_index]
            clf=SVC(C=C,degree=deg,kernel='poly')
            clf.fit(X_train_k,y_train_k)
            y_pred_rbf=clf.predict(X_test_k)
            scores_BDT.append(accuracy_score(y_test_k,y_pred_rbf))
        print  C,",",deg , "," , np.mean(scores_BDT)

print "SVM sigmoid KERNEL"
print "C_Value ","gamma", "Cross Validation Accuracy " 
for C in C_2d_range:
    for gamma in gamma_2d_range:
        scores_BDT=list()
        for train_index,test_index in kf:
            X_train_k,X_test_k=X[train_index],X[test_index]
            y_train_k,y_test_k=y[train_index],y[test_index]
            clf=SVC(C=C,gamma=gamma,kernel='sigmoid')
            clf.fit(X_train_k,y_train_k)
            y_pred_rbf=clf.predict(X_test_k)
            scores_BDT.append(accuracy_score(y_test_k,y_pred_rbf))
        print  C,",",gamma , "," , np.mean(scores_BDT) 
        
       
for train_index,test_index in kf:
    X_train_k,X_test_k=X[train_index],X[test_index]
    y_train_k,y_test_k=y[train_index],y[test_index]
    clf=SVC()
    clf.fit(X_train_k,y_train_k)
    y_pred_rbf=clf.predict(X_test_k)
    print "SVM Accuracy : " ,accuracy_score(y_test_k,y_pred_rbf)
    k_model=BoostedDT(numBoostingIters=100,maxTreeDepth=3)
    k_model.fit(X_train_k,y_train_k)
    y_pred_k=k_model.predict(X_test_k)
    print "BoostedDT : ",accuracy_score(y_test_k,y_pred_k)
    scores.append(accuracy_score(y_test_k,y_pred_k))
    scores_rbf.append(accuracy_score(y_test_k,y_pred_rbf))
acc=0
acc_rbf=0

for i in range(len(scores)):
    acc=scores[i]+acc
    acc_rbf=scores_rbf[i]+acc_rbf



# output predictions on the remaining data
ypred_DT = modelDT.predict(Xtest)
ypred_BoostedDT = modelBoostedDT.predict(Xtest)
y_pred_SK=model.predict(Xtest)

# compute the training accuracy of the model
accuracyDT = accuracy_score(ytest, ypred_DT)
accuracy_SK=accuracy_score(ytest,y_pred_SK)

accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)
accuracy_rbf = accuracy_score(ytest, y_pred_rbf1)

print "Decision Tree Accuracy = "+str(accuracyDT)
print "SVC RBF Accuracy = "+str(accuracy_rbf)
print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)
print "Cross validation acccuracy BoostedDT= "+str(acc/len(scores))
print "Cross Validation for AdaBoostClassifier Sklearn= "+str(results.mean())
print "Adaboost sklearn acccuracy= "+str(accuracy_SK)
print "Cross Validation for SVC RBF sklearn acccuracy= "+str(acc_rbf/len(scores))
'''