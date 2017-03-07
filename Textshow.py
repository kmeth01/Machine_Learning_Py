import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp

categories = ['comp.sys.mac.hardware', 'rec.motorcycles','comp.graphics', 'sci.space','talk.politics.mideast']

twenty_train=fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
count_vect = CountVectorizer(analyzer = 'word',stop_words='english')
twenty_test=fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, shuffle=True, random_state=42)
X_train=count_vect.fit_transform(twenty_train.data)

X_test = count_vect.transform(twenty_test.data)

tfidf_transformer = TfidfTransformer(norm='l2',sublinear_tf = True)
tfidf_Xtrain = tfidf_transformer.fit_transform(X_train)
tfidf_Xtest = tfidf_transformer.transform(X_test)

y = twenty_train.target
K = np.unique(twenty_train.target)
y = label_binarize(y, classes=K)
y_train = y

cls = y.shape[1]

y_test = twenty_test.target
y_test = label_binarize(y_test, classes=K) 

random_state=np.random.RandomState(0)

# Learn to predict each class against the other
classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

n_classes = 5

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    
lw=2

classifier = OneVsRestClassifier(MultinomialNB())
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

for i in range(n_classes):
    fpr[i+5], tpr[i+5], _ = roc_curve(y_test[:, i], y_score[:, i])

n_classes = 10
Methods=['MultinomialNB','SVM Classifier']    
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=('ROC curve of' + Methods[i/n_classes]+ K[i%n_classes]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()