#Taskoudis Dimitris


import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn import datasets, metrics
from sklearn import decomposition
import time 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import  KernelPCA
from sklearn.utils import shuffle

#load data
mnist = fetch_mldata("MNIST original")
#Import labels
X = np.float32(mnist.data[:70000])/ 255.
y = np.float32(mnist.target[:70000])
X,y = shuffle(X,y)

#split the data
X_train = np.float32(X[:15000])/255.
y_train =  np.float32(y[:15000])
X_test =  np.float32(X[60000:])/ 255.
y_test = np.float32(y[60000:])

#import PCA for variance
pca = PCA(n_components=100)
principalComponents = pca.fit_transform(X)
pca.explained_variance_ratio_
print ("explained_variance is ", pca.explained_variance_ratio_) 

#import kPCA
kpca = KernelPCA(kernel="rbf",n_components=100 , gamma=0.01)
X_kpca = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


#import LDA for components = classes - 1 
lda = LDA()
X_lda = lda.fit_transform(X_kpca,y_train)
X_test = lda.transform(X_test)


#calculation time fitting for KNN
start = int(round(time.time() * 1000))

#import KNN
clf = neighbors.KNeighborsClassifier(n_neighbors=15)
clf.fit(X_lda, y_train)
print (clf)

end = int(round(time.time() * 1000))
print("--KNN fitting finished in ", (end-start), "ms")

expected = y_test
predicted = clf.predict(X_test) 

#import report, confusion matrix for results
print(clf.score(X_test, y_test))
print("Classification report for kNN classifier %s:\n%s\n"
     % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print (clf.score(X_test,y_test))

#calculation time fitting for Nearest Centroid
start = int(round(time.time() * 1000))

#Import Nearest Centroid
classifier = NearestCentroid()
classifier.fit(X_lda, y_train)
NearestCentroid(metric='euclidean', shrink_threshold=None)
print (classifier)

end = int(round(time.time() * 1000))
print("--Centroid fitting finished in ", (end-start), "ms")

expected = y_test
predicted = classifier.predict(X_test)

#import report,confusion matrix for results
print(classifier.score(X_test, y_test))
print("Classification report for Centroid classifier %s:\n%s\n"
     % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print (classifier.score(X_test,y_test))






