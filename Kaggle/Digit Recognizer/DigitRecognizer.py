import pandas as pd
import csv

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")

y = train['label']
X = train.drop(['label'],axis=1)

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

Max_components = 30
Start_compoent = 20
PCAscore = list()
for i in [n+20 for n in range(Max_components)]:
	print('Start to calculation...%d' % i)
	pca = PCA(n_components=i,whiten=True)
	pca.fit(X)
	train_data = pca.transform(X)
	X_train, X_test, y_train, y_test = train_test_split(train_data,y,test_size=0.1,random_state=42)
	classifier = svm.SVC()
	classifier.fit(X_train,y_train)
	PCAscore.append(classifier.score(X_test,y_test))

print("Max PCA Score: %f" % max(PCAscore))

best_component = PCAscore.index(max(PCAscore))+Start_compoent
pca = PCA(n_components=best_component,whiten=True)
pca.fit(X)
train_data = pca.transform(X)
classifier = svm.SVC()
classifier.fit(train_data,y)
predictedlabel = classifier.predict(pca.transform(test))

result = pd.DataFrame()
result['ImageId'] = [n+1 for n in range(len(predictedlabel))]
result['Label'] = predictedlabel
result.to_csv('submission.csv')