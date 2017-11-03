import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import roc_curve, precision_recall_curve,classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_circles,make_moons,make_friedman1,make_s_curve,make_gaussian_quantiles,make_swiss_roll


n_samples = 400
noise = 0.1
distance = 1.0

classifier = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(20, 10), random_state=1)

# Normal Dataset
X = np.zeros((2*n_samples,2))
y = np.zeros(2*n_samples)
X[:n_samples,0] = np.random.normal(loc=distance/2, scale=0.5, size=[n_samples])
X[:n_samples,1] = np.random.normal(loc=-distance/2, scale=0.5, size=[n_samples])
X[n_samples:,0] = np.random.normal(loc=-distance/2, scale=0.5, size=[n_samples])
X[n_samples:,1] = np.random.normal(loc=distance/2, scale=0.5, size=[n_samples])
y[n_samples:] = np.ones(n_samples)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
classifier = MLPClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
plt.figure()
x0 = np.linspace(-2,2,100)
[xv,yv] = np.meshgrid(x0,x0)
plotdata = classifier.predict([[x0[i],x0[j]] for i in range(100) for j in range(100)]).reshape(xv.shape)
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

# Circle Dataset
[X,y] = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
plt.figure()
x0 = np.linspace(-2,2,100)
[xv,yv] = np.meshgrid(x0,x0)
plotdata = classifier.predict([[x0[i],x0[j]] for i in range(100) for j in range(100)]).reshape(xv.shape)
plt.pcolormesh(xv,yv,plotdata)
ax=plt.gca()
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_yticklabels([-2, -1, 0, 1, 2])
plt.colorbar()
plt.scatter(X[:,0],X[:,1],s=10, c=y, cmap=plt.cm.Spectral)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Dataset')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# Circle Dataset
[X,y] = make_moons(n_samples=n_samples, noise=noise)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
plt.figure()
x0 = np.linspace(-2,2,100)
[xv,yv] = np.meshgrid(x0,x0)
plotdata = classifier.predict([[x0[i],x0[j]] for i in range(100) for j in range(100)]).reshape(xv.shape)
plt.pcolormesh(xv,yv,plotdata)
ax=plt.gca()
ax.set_yticks([-2, -1, 0, 1, 2])
ax.set_yticklabels([-2, -1, 0, 1, 2])
plt.colorbar()
plt.scatter(X[:,0],X[:,1],s=10, c=y, cmap=plt.cm.Spectral)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('Dataset')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

plt.show()