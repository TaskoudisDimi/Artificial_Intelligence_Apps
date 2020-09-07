import numpy as np
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import decomposition
import time 

#load data from sclearn
mnist = fetch_mldata("MNIST original")
X = mnist.data
y = mnist.target

#Values on labels
for count,i in enumerate(y):
    if i%2==0:
       y[count]=0
    else:
        y[count]=1

# train data = 60.000
X_train = np.float32(mnist.data[:60000])/ 255.
y_train =  np.float32(mnist.target[:60000])

#Finding components through PCA, variance explained
pca = decomposition.PCA(n_components = 100, svd_solver='full')
pca.fit(X_train) 
print(np.sum(pca.explained_variance_ratio_))


# Fitting the new dimensions to the train-set.
train_ext = pca.fit_transform(X_train)

# calculation time fitting for SVM
start = int(round(time.time() * 1000))

#import SVM
classifier = svm.SVC(gamma = 0.1, C=2, kernel='rbf')
classifier.fit(train_ext,y_train)

end = int(round(time.time() * 1000))
print("--SVM fitting finished in ", (end-start), "ms")

# test data = 10.000
X_test, y_test = np.float32(mnist.data[60000:]) / 255., np.float32(mnist.target[60000:])

# Fitting the new dimensions.
test_ext = pca.transform(X_test)

expected = y_test
predicted = classifier.predict(test_ext)

#report for results and confusion matrix
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

#illustrate some right and wrong predictions
for i in np.random.choice(np.arange(0, len(expected)), size = (3,)):
    pred = classifier.predict(np.atleast_2d(test_ext[i]))	
    image = (X_test[i] * 255).reshape((28, 28)).astype("uint8")	
    plt.figure()  
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Actual digit is {0}, predicted {1}".format(expected[i], pred[0]))
plt.show()







