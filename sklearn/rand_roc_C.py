import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, precision_recall_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

distance = 0.25

n_samples = 300
X = np.zeros((2*n_samples,2))
y = np.zeros(2*n_samples)
X[:n_samples,0] = np.random.normal(loc=distance/2, scale=0.5, size=[n_samples])
X[:n_samples,1] = np.random.normal(loc=-distance/2, scale=0.5, size=[n_samples])
X[n_samples:,0] = np.random.normal(loc=-distance/2, scale=0.5, size=[n_samples])
X[n_samples:,1] = np.random.normal(loc=distance/2, scale=0.5, size=[n_samples])
y[n_samples:] = np.ones(n_samples)

plt.figure()

plt.scatter(X[:,0],X[:,1],s=40, c=y, cmap=plt.cm.Spectral)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Dataset')


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

classifiers = [
    LogisticRegression(max_iter=1), 
    LogisticRegression(max_iter=2),
    LogisticRegression(max_iter=100)]

# ROC
plt.figure()

colors = ['b','r','g']
for i in range(len(classifiers)):
    y_score = classifiers[i].fit(X_train,y_train).decision_function(X_test)
    FPR,TPR,_ = roc_curve(y_test,y_score)
    plt.plot(FPR,TPR,color=colors[i],label='ROC with classifiers {num}'.format(num=i))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

if 0:
    # PR
    plt.figure()

    colors = ['b','r','g']
    for i in range(len(classifiers)):
        y_score = classifiers[i].fit(X_train,y_train).decision_function(X_test)
        P, R, _= precision_recall_curve(y_test,y_score)
        plt.plot(R,P,color=colors[i],label='ROC with classifiers {num}'.format(num=i))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R')
    plt.legend(loc="lower right")

plt.show()