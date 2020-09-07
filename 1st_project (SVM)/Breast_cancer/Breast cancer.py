#Taskoudis Dimitris

import pandas as pd
from sklearn.neighbors import NearestCentroid
from sklearn import neighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
import datetime

#import data
data = pd.read_csv("C:/Users/ASUS/Desktop/Εργασία Τεχνητής 'Ορασης/data.csv")
#labels
X = data.iloc[:, 2:-1]
y = data.iloc[:, 1]
y = [1 if i=='M' else 0 for i in y]

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
##scale data
#scaler = MinMaxScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.fit_transform(X_test)

# find biggest difference between min value and any point of dataset
min_train = X_train.min()
range_train = (X_train-min_train).max() 
X_train= (X_train - min_train)/range_train
min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test= (X_test - min_test)/range_test

#import PCA
pca = decomposition.PCA(n_components = 2, svd_solver='full')
pca.fit(X_train) 


#time measurement
now = datetime.datetime.now()
#NearestCentoid
#clf = NearestCentroid(shrink_threshold= None)
#KNN
#clf = neighbors.KNeighborsClassifier(n_neighbors = 15, weights='uniform') 
#SVM
clf = SVC()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
parameters = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
                {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['sigmoid']},
                {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
                {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['poly']}]
grid = GridSearchCV(SVC(), parameters, verbose= 4, refit=True)
grid.fit(X_train, y_train)
grid.best_params_
print (grid.best_params_)

then = datetime.datetime.now()
diff = then - now
print (diff)

optimized_preds = clf.predict(X_test)
cm = confusion_matrix(y_test, optimized_preds)
print (cm)

print(classification_report(y_test, y_predict))
print(clf.score(X_train, y_train), clf.score(X_test, y_test))
print(grid.score(X_train, y_train), grid.score(X_test, y_test))



