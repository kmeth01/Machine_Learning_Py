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
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from scipy import interp
import time
import matplotlib.pyplot as plt
from itertools import cycle

categories = ['comp.sys.mac.hardware', 'rec.motorcycles','comp.graphics', 'sci.space','talk.politics.mideast']

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
count_vect = CountVectorizer(analyzer = 'word',stop_words='english', ngram_range = (1,1))
newsgroups_test=fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
X_train_counts=count_vect.fit_transform(twenty_train.data)
test_tfidf_transformer = TfidfTransformer(norm='l2',sublinear_tf = True )
Norm_X_train_tfidf = test_tfidf_transformer.fit_transform(X_train_counts)

#print Norm_X_train_tfidf[1,:].sum()
#print "Next "

No_Norm_tfidf_transformer = TfidfTransformer()
No_Norm_X_train_tfidf = No_Norm_tfidf_transformer.fit_transform(X_train_counts)

#print No_Norm_X_train_tfidf[1,:].sum()

# clf=MultinomialNB().fit(X_train.target)
Execution_times=list()
Test_data=newsgroups_test.data
start_time=time.time()
text_clf=Pipeline([('vect',CountVectorizer(analyzer = 'word',stop_words='english')),('tfidf',TfidfTransformer(norm='l2',sublinear_tf = True,smooth_idf=False)),('clf',MultinomialNB())])
text_clf=text_clf.fit(twenty_train.data,twenty_train.target)
Training_time=time.time()-start_time
Execution_times.append(Training_time)
predicted=text_clf.predict(Test_data)
train_predicted=text_clf.predict(twenty_train.data)

accuracy_score_NB=list()
Precision_NB=list()
Recall_NB=list()

accuracy_score_NB.append(metrics.accuracy_score(newsgroups_test.target,predicted))
accuracy_score_NB.append(metrics.accuracy_score(twenty_train.target,train_predicted))

Precision_NB.append(metrics.precision_score(newsgroups_test.target,predicted,average='weighted'))
Precision_NB.append(metrics.precision_score(twenty_train.target,train_predicted,average='weighted'))

Recall_NB.append(metrics.recall_score(newsgroups_test.target,predicted,average='weighted'))
Recall_NB.append(metrics.recall_score(twenty_train.target,train_predicted,average='weighted'))

print "Metrics Table for MultinomialNB"
print "data","\taccuracy","\tprecision","\trecall","\t\tTraining Time"
print "Train","\t",accuracy_score_NB[1],"\t",Precision_NB[1],"\t",Recall_NB[1],"\t",Execution_times[0]
print "Test","\t",accuracy_score_NB[0],"\t",Precision_NB[0],"\t",Recall_NB[0]
# ### SVM Cosine_Similarility Classifier###
# parameters = {'vect__ngram_range': [(1, 1)],'tfidf__use_idf': [True],'cld__C': [0.2,0.4,0.6,0.8,1,1.5]}
# text_cld=Pipeline([('vect',CountVectorizer(analyzer = 'word',stop_words='english')),('tfidf',TfidfTransformer(norm='l2',sublinear_tf = True,smooth_idf=False)),('cld',SVC(C=1.0,kernel=cosine_similarity))])

# gs_clf = GridSearchCV(text_cld, parameters)
# gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)

# print "GridSearchCV Best Score : ",gs_clf.best_score_ 
# for param_name in sorted(parameters.keys()):
  # print ("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
start_time=time.time() 
text_cld=Pipeline([('vect',CountVectorizer(analyzer = 'word',stop_words='english')),('tfidf',TfidfTransformer(norm='l2',sublinear_tf = True,smooth_idf=False)),('cld',SVC(C=1.5,kernel=cosine_similarity))])
text_cld=text_cld.fit(twenty_train.data,twenty_train.target)
Training_time=time.time()-start_time
predicted=text_cld.predict(Test_data)
predicted_train=text_cld.predict(twenty_train.data)
#print "SVM cosine_similarity Classifier Best: ",metrics.accuracy_score(newsgroups_test.target,predicted)
Execution_times.append(Training_time)
predicted=text_cld.predict(Test_data)
train_predicted=text_cld.predict(twenty_train.data)

accuracy_score_SVM=list()
Precision_SVM=list()
Recall_SVM=list()

accuracy_score_SVM.append(metrics.accuracy_score(newsgroups_test.target,predicted))
accuracy_score_SVM.append(metrics.accuracy_score(twenty_train.target,train_predicted))

Precision_SVM.append(metrics.precision_score(newsgroups_test.target,predicted,average='weighted'))
Precision_SVM.append(metrics.precision_score(twenty_train.target,train_predicted,average='weighted'))

Recall_SVM.append(metrics.recall_score(newsgroups_test.target,predicted,average='weighted'))
Recall_SVM.append(metrics.recall_score(twenty_train.target,train_predicted,average='weighted'))

print "Metrics Table for SVM with Cosine_Similarility"
print "data","\taccuracy","\tprecision","\trecall","\t\tTraining Time"
print "Train","\t",accuracy_score_SVM[1],"\t",Precision_SVM[1],"\t",Recall_SVM[1],"\t",Execution_times[1]
print "Test","\t",accuracy_score_SVM[0],"\t",Precision_SVM[0],"\t",Recall_SVM[0]

# #### SGD Classifier###
# text_cld=Pipeline([('vect',CountVectorizer(analyzer = 'word',stop_words='english')),('tfidf',TfidfTransformer(norm='l2',sublinear_tf = True,smooth_idf=False)),('cld',SGDClassifier(loss='hinge', penalty='l2',alpha=0.0001, n_iter=20, random_state=42))])
# text_cld=text_cld.fit(twenty_train.data,twenty_train.target)
# predicted=text_cld.predict(Test_data)

# print "SGD Classifier Best: ",metrics.accuracy_score(newsgroups_test.target,predicted)
# print (metrics.classification_report(newsgroups_test.target, predicted,target_names=newsgroups_test.target_names))

##########Below Code Plots the Curves for specific categories #####################
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
classifier = OneVsRestClassifier(SVC(C=1.5,kernel=cosine_similarity))
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
Methods=['SVM','NB']    
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink', 'green'])
for i, color in zip(range(n_classes), colors):
    #print K[i]
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label=Methods[i/5]+" "+'ROC curve of class' + categories[i%5].split(".")[-1])

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for SVM and MultinomialNB')

plt.legend(loc="lower right")
plt.savefig("graphTextClassifierROC.pdf")