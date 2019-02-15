#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the Dataset
dataset = pd.read_csv("Mall_Customers.csv")
x = dataset.iloc[:, [3,4]].values

#Using Elbow Method to find optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters = i, init = "k-means++", n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.label("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()  

#Applying KMeans to dataset
kmeans = KMeans(n_clusters = 5, init = "k-means++", n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#Visualising the results
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1], s=100, c="red", label = "Careful")
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1], s=100, c="blue", label = "Standard")
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1], s=100, c="pink", label = "Target")
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1], s=100, c="orange", label = "Careless")
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1], s=100, c="violet", label = "Sensible")
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c="yellow", label = "Centroid")
plt.title("Cluster of Clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()  