#Taskoudis Dimitris


from sklearn import datasets
from sklearn.neighbors import NearestCentroid
from sklearn import neighbors
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target
#values on labels
for count,i in enumerate(y):
    if i%2==0:
       y[count]=0
    else:
        y[count]=1
        
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0, stratify=y)

#scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#time measurement
now = datetime.datetime.now()
#NearestCentoid
#clf = NearestCentroid(shrink_threshold= 0.5)
#KNN
#clf = neighbors.KNeighborsClassifier(n_neighbors = 5, weights='uniform') 
#SVM
clf = SVC()
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
parameters = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']}, 
              {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['sigmoid']},
              {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']},
              {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['poly']}]
grid = GridSearchCV(SVC(), parameters, verbose = 4, refit=True)
grid.fit(X_train, y_train)
grid.best_params_
print (grid.best_params_)
then = datetime.datetime.now()
diff = then - now
print (diff)

cm = confusion_matrix(y_test, y_predicted)
print(grid.score(X_train, y_train), grid.score(X_test, y_test))
print(classification_report(y_test, y_predicted))
print (cm)

    
