import numpy as np 
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Helvetica Light' # 'Helvetica'
rcParams['font.size'] = 18
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, precision_recall_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

distance = 1.0

n_samples = 300
X = np.zeros((2*n_samples,2))
y = np.zeros(2*n_samples)
X[:n_samples,0] = np.random.normal(loc=distance/2, scale=0.5, size=[n_samples])
X[:n_samples,1] = np.random.normal(loc=-distance/2, scale=0.5, size=[n_samples])
X[n_samples:,0] = np.random.normal(loc=-distance/2, scale=0.5, size=[n_samples])
X[n_samples:,1] = np.random.normal(loc=distance/2, scale=0.5, size=[n_samples])
y[n_samples:] = np.ones(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

classifier = LinearDiscriminantAnalysis()

y_score = classifier.fit(X_train,y_train).decision_function(X_test)
y_pred = classifier.predict(X_test)

#print(classification_report(y_test, y_pred))

plt.figure()

x0 = np.linspace(-2,2,100)
[xv,yv] = np.meshgrid(x0,x0)
plotdata = classifier.predict([[x0[i],x0[j]] for i in range(100) for j in range(100)]).reshape(xv.shape)
#plotdata = np.reshape(plotdata,100,100)

plt.pcolormesh(xv,yv,plotdata)
ax=plt.gca()
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_yticklabels([-2, -1, 0, 1, 2])

plt.colorbar()
plt.scatter(X[:,0],X[:,1],s=40, c=y, cmap=plt.cm.Spectral)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Dataset')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# ROC
plt.figure()

FPR,TPR,_ = roc_curve(y_test,y_score)
plt.plot(FPR,TPR,color='darkorange',label='ROC')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")

# PR
plt.figure()

P, R, _= precision_recall_curve(y_test,y_score)
plt.plot(R,P,color='darkorange',label='P-R')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('P-R')
plt.legend(loc="lower right")
plt.show()