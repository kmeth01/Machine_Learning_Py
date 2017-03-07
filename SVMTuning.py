"""
======================
Script to Explore SVMs
======================

Simple script to explore SVM training with varying C

Example adapted from scikit_learn documentation by Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# load the data
filename = 'data/svmTuningData.dat'
allData = np.loadtxt(filename, delimiter=',')

X = allData[:,:-1]
Y = allData[:,-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=0)
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2,1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]
                     
scores = list()


C_List = list()
for i in range(1,1000,5):
    C_List.append(i)
_gaussSigma_List=np.arange(0.01,20,0.03)

for C in C_List:
    for _gaussSigma in _gaussSigma_List:
        equivalentGamma = 1.0 / (2 * _gaussSigma ** 2)
        #model = svm.SVC(C = C, kernel='rbf', gamma=equivalentGamma)
        # train the SVM
        #print "Training the SVM"
        clf = SVC(C = C, kernel='rbf', gamma=equivalentGamma)
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        accur=accuracy_score(y_test,y_pred)
        scores.append([accur,C,_gaussSigma])
Max=scores[0][0]
MaxIndex=0
for i in range(len(scores)):
    if scores[i][0]>Max:
       Max=scores[i][0]
       MaxIndex=i

       
#print ""

# for score in scores:
    # print("# Tuning hyper-parameters for %s" % score)
    # print()

    # clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       # scoring='%s_weighted' % score)
    # clf.fit(X_train, y_train)

    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # for params, mean_score, scores in clf.grid_scores_:
        # print("%0.3f (+/-%0.03f) for %r"
              # % (mean_score, scores.std() * 2, params))
    # print()

    # print("Detailed classification report:")
    # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # print()
    # y_true, y_pred = y_test, clf.predict(X_test)
    # print(classification_report(y_true, y_pred))
    # print()
equivalentGamma = 1.0 / (2 * scores[MaxIndex][2] ** 2)
model = SVC(C = scores[MaxIndex][1] , kernel='rbf', gamma=equivalentGamma)
model.fit(X, Y)
print scores[MaxIndex]

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('SVM decision surface with C = '+str(C))
plt.axis('tight')
plt.show()
