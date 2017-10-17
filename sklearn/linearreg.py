import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, precision_recall_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression

k = 1.0
xmin,xmax = 0,10
delta = 1

n_samples = 100
X = np.zeros((n_samples,1))
X[:,0] = np.linspace(xmin,xmax,n_samples)
y = X + np.random.normal(loc=0, scale=delta, size=(n_samples,1)) #+ 2*np.sin(2*np.pi*X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

classifier = LinearRegression().fit(X_train,y_train)

y_pred = classifier.predict(X_test)

plt.figure()

plt.scatter(X,y,color='darkorange',label='Dataset')
plt.plot(X_test,y_pred,label='Linear model')
plt.xlim([X.min(), X.max()])
plt.ylim([y.min(), y.max()])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend(loc="lower right")

plt.show()