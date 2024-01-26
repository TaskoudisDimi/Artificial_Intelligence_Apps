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



###########################################
# from keras.datasets import mnist
# from sklearn import metrics
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# import numpy as np


# x_train = x_train.astype('float32') 
# x_test = x_test.astype('float32') 
# # Normalization
# x_train = x_train/255.0
# x_test = x_test/255.0



# X_train = x_train.reshape(len(x_train),-1)
# X_test = x_test.reshape(len(x_test),-1)


# from sklearn.cluster import MiniBatchKMeans
# total_clusters = len(np.unique(y_test))
# # Initialize the K-Means model
# kmeans = MiniBatchKMeans(n_clusters = total_clusters)
# # Fitting the model to training set
# kmeans.fit(X_train)

# def retrieve_info(cluster_labels,y_train):
#     reference_labels = {}
#     # For loop to run through each label of cluster label
#     for i in range(len(np.unique(kmeans.labels_))):
#         index = np.where(cluster_labels == i,1,0)
#         num = np.bincount(y_train[index==1]).argmax()
#         reference_labels[i] = num
#     return reference_labels


# reference_labels = retrieve_info(kmeans.labels_,y_train)
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#   number_labels[i] = reference_labels[kmeans.labels_[i]]

# print(number_labels[:20].astype('int'))
# print(y_train[:20])

# from sklearn.metrics import accuracy_score
# print(accuracy_score(number_labels,y_train))

# # Function to calculate metrics for the model
# def calculate_metrics(model,output):
#  print("Number of clusters is {}'".format(model.n_clusters))
#  print("Inertia : {}".format(model.inertia_))
#  print("Homogeneity :       {}".format(metrics.homogeneity_score(output,model.labels_)))


# from sklearn import metrics
# cluster_number = [10,16,36,64,144,256]
# for i in cluster_number:
#     total_clusters = len(np.unique(y_test))

#     kmeans = MiniBatchKMeans(n_clusters = i)

#     kmeans.fit(X_train)

 
# calculate_metrics(kmeans,y_train)
# # Calculating reference_labels
# reference_labels = retrieve_info(kmeans.labels_,y_train)
# # ‘number_labels’ is a list which denotes the number displayed in image
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
 
#  number_labels[i] = reference_labels[kmeans.labels_[i]]
 
# print("Accuracy score : {}".format(accuracy_score(number_labels,y_train)))



# # Testing model on Testing set
# # Initialize the K-Means model
# kmeans = MiniBatchKMeans(n_clusters = 256)
# # Fitting the model to testing set
# kmeans.fit(X_test)
# # Calculating the metrics
# calculate_metrics(kmeans,y_test)
# # Calculating the reference_labels
# reference_labels = retrieve_info(kmeans.labels_,y_test)
# # ‘number_labels’ is a list which denotes the number displayed in image
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
 
#  number_labels[i] = reference_labels[kmeans.labels_[i]]
 
# print("Accuracy score : {}".format(accuracy_score(number_labels,y_test)))


# # Cluster centroids is stored in ‘centroids’
# centroids = kmeans.cluster_centers_

# centroids = centroids.reshape(256,28,28)

# centroids = centroids * 255





# # RGB image is converted to Monochrome image
# from skimage import color
# from skimage import io
# image = color.rgb2gray(io.imread("C:/Users/chris/Desktop/Dimitris/Tutorials/AI/Computational-Intelligence-and-Statistical-Learning/TrainedModels/1st_project (SVM)/Mnist/6.png"))

# # Reshaping into a row vector
# image = image.reshape(1,28*28)
# # Importing the dataset from keras
# from keras.datasets import mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # Normalization of ‘x_train’
# x_train = x_train.astype('float32')
# x_train = x_train/255.0
# x_train = x_train.reshape(60000,28*28)

# # Training the model
# kmeans = MiniBatchKMeans(n_clusters=256)
# kmeans.fit(x_train)


# reference_labels = retrieve_info(kmeans.labels_,y_train)
# number_labels = np.random.rand(len(kmeans.labels_))
# for i in range(len(kmeans.labels_)):
#   number_labels[i] = reference_labels[kmeans.labels_[i]]


# predicted_cluster = kmeans.predict(image)

# print(number_labels[[predicted_cluster]])


