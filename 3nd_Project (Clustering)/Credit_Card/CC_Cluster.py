import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import SpectralClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.metrics import silhouette_score 
from sklearn import manifold
import time


#Import data
data = pd.read_csv("C:/Users/ASUS/Desktop/Εργασία Τεχνητής 'Ορασης/Credit_Card.csv")
#drop CUST_ID from data
data = data.drop('CUST_ID', axis = 1)
#filling the vacancies 
data.fillna(method ='ffill', inplace = True)


# Scaling the Data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(data) 
  
# Normalizing the Data 
X_normalized = normalize(X_scaled) 
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 
  

#Time calculation 
start = int(round(time.time() * 1000))
#Import Spectral Embedding
X_spec = manifold.SpectralEmbedding(n_components=2, affinity='nearest_neighbors', gamma=None, random_state=None, eigen_solver=None, n_neighbors=15).fit_transform(X_normalized)
end = int(round(time.time() * 1000))
print("Time for spectral embedding is",(end-start), "ms")

##Time calculation
#start = int(round(time.time() * 1000))
##Import Isomap
#X_iso = manifold.Isomap(n_neighbors=15, n_components=2).fit_transform(X_normalized)
#end = int(round(time.time() * 1000))
#print("Time for isomap is", (end-start), "ms")
    
# Building the clustering model
spectral_model_nn = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors') 
  
# Training the model and Storing the predicted cluster labels 
#y_pred = spectral_model_nn.fit_predict(X_iso)
y_pred = spectral_model_nn.fit_predict(X_spec)
  
# List of Silhouette Scores 
s_scores = [] 
  
# Evaluating the performance  
s_scores.append(silhouette_score(data, y_pred)) 
print(s_scores)

#Plot the data in 2 dimensions after reducing and clustering
plt.title("Spectral Embedding and Spectral Clustering for data Credit_Card (2 clusters)")
#plt.title("Isomap and Spectral Clustering for data Credit_Card (2 clusters)")
plt.scatter(X_spec[:, 0], X_spec[:, 1], c=y_pred, s=50, cmap=plt.cm.get_cmap("jet", 2))
plt.colorbar(ticks=range(2))
plt.clim(-0.5,1.5)
plt.show()


