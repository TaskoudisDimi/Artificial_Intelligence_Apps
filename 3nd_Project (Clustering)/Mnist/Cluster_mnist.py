from sklearn.metrics.cluster import completeness_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import manifold
import time
from sklearn import cluster
import pandas as pd 
from sklearn.model_selection import train_test_split

#Import data
data = pd.read_csv("C:/Users/ASUS/Desktop/Εργασία Τεχνητής 'Ορασης/mnist_data.csv")

#Import 'label' 
y = data['label']
## Drop the 'label' from data 
X = data.drop(columns = 'label')

#Normalized data
X = X/255.0
    
# import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=0)

#Time calculation 
start = int(round(time.time() * 1000))
#Import Spectral Embedding
X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=5).fit_transform(X_train)
end = int(round(time.time() * 1000))
print("Time for spectral embedding is",(end-start), "ms")

##Time calculation
#start = int(round(time.time() * 1000))
##Import Isomap
#X_iso = manifold.Isomap(n_neighbors=5, n_components=2).fit_transform(X_train)
#end = int(round(time.time() * 1000))
#print("Time for isomap is", (end-start), "ms")


#Building the clustering model
spectral = cluster.SpectralClustering(n_clusters=10, affinity="nearest_neighbors")

#X = spectral.fit(X_iso)
X = spectral.fit(X_spec)

#Training the model and Storing the predicted cluster labels 
#y_pred = spectral.fit_predict(X_iso)
y_pred = spectral.fit_predict(X_spec)

# clustering evaluation metrics
print(confusion_matrix(y_train, y_pred))
print (completeness_score(y_train, y_pred))


#Plot the data in 2 dimensions after reducing and clustering
plt.title("Spectral Embedding and Spectral Clustering for data MNIST (1 clusters)")
#plt.title("Isomap and Spectral Clustering for data MNIST (10 clusters)")
plt.scatter(X_spec[:, 0], X_spec[:, 1], c=y_pred, s=50, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()