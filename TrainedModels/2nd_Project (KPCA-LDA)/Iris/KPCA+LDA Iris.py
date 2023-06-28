#Taskoudis Dimitris


from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import metrics
import time 
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import  KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import MinMaxScaler


#load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0, stratify=y)

#scale data between 0,1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#import PCA for explained variance
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
pca.explained_variance_ratio_
print (pca.explained_variance_ratio_) 

#import kPCA
kpca = KernelPCA(kernel="rbf", n_components = 2 , gamma = 0.01)
X_train = kpca.fit_transform(X_train_scaled)
X_test = kpca.transform(X_test_scaled)

#import LDA with components = classes-1
lda = LDA()
X_lda = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)

#calculation time fitting for Nearest Centroid
start = int(round(time.time() * 1000))

#import Nearest Centroid
clf = NearestCentroid(metric='euclidean', shrink_threshold= None)
clf.fit(X_lda, y_train)

end = int(round(time.time() * 1000))
print("--NC fitting finished in ", (end-start), "ms")

expected = y_test
predicted = clf.predict(X_test)

#report for results 
print("Classification report for kNN classifier %s:\n%s\n"% (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print (clf.score(X_test,y_test))

#import KNN
clf = neighbors.KNeighborsClassifier(n_neighbors = 15)
clf.fit(X_lda, y_train)

end = int(round(time.time() * 1000))
print("--KNN fitting finished in ", (end-start), "ms")

expected = y_test
predicted = clf.predict(X_test)

#report for results
print("Classification report for kNN classifier %s:\n%s\n"% (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
print (clf.score(X_test,y_test))




