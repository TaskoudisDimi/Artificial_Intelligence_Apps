import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn import decomposition
import time 
from sklearn.datasets import fetch_openml

# Load data from OpenML repository
mnist = fetch_openml(name='mnist_784', version=1)

# Extract features and labels
X, y = mnist['data'], mnist['target']

# Convert labels to binary (even=0, odd=1)
y = np.array([int(label) % 2 for label in y])

# train data = 60,000
X_train, y_train = X[:60000] / 255.0, y[:60000]

# Finding components through PCA, variance explained
pca = decomposition.PCA(n_components=100, svd_solver='full')
pca.fit(X_train)
print(np.sum(pca.explained_variance_ratio_))

# Fitting the new dimensions to the train-set.
train_ext = pca.transform(X_train)

# calculation time fitting for SVM
start = int(round(time.time() * 1000))

# Import SVM
classifier = svm.SVC(gamma=0.1, C=2, kernel='rbf')
classifier.fit(train_ext, y_train)

end = int(round(time.time() * 1000))
print("--SVM fitting finished in ", (end-start), "ms")

# test data = 10,000
X_test, y_test = X[60000:] / 255.0, y[60000:]

# Fitting the new dimensions.
test_ext = pca.transform(X_test)

expected = y_test
predicted = classifier.predict(test_ext)

# Report for results and confusion matrix
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

# Illustrate some right and wrong predictions
for i in np.random.choice(np.arange(0, len(expected)), size=(3,)):
    pred = classifier.predict(np.atleast_2d(test_ext[i]))
    image = (X_test[i] * 255).reshape((28, 28)).astype("uint8")
    plt.figure()
    plt.axis("off")
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Actual digit is {0}, predicted {1}".format(expected[i], pred[0]))
plt.show()
