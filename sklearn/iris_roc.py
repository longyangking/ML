import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import datasets
from sklearn.metrics import roc_curve, precision_recall_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()
X = iris.data 
y = iris.target

y = label_binarize(y,classes=[0,1,2])
n_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)

classifier = OneVsRestClassifier(LogisticRegression())

y_score = classifier.fit(X_train,y_train).decision_function(X_test)
y_pred = classifier.predict(X_test)

print(classification_report(y_test, y_pred,target_names=iris.target_names))

# ROC
plt.figure()
colors = ['aqua', 'darkorange', 'cornflowerblue']
for i in range(n_classes):
    FPR,TPR,_ = roc_curve(y_test[:,i],y_score[:,i])
    plt.plot(FPR,TPR,color=colors[i],
            label='ROC: {name}'.format(name=iris.target_names[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

# PR
plt.figure()

for i in range(n_classes):
    P, R, _= precision_recall_curve(y_test[:,i],y_score[:,i])
    plt.plot(R,P,color=colors[i],
            label='P-R: {name}'.format(name=iris.target_names[i]))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR')
plt.legend(loc="lower right")
plt.show()


