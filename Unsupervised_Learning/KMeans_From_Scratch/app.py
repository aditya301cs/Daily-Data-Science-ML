# ---------------------------------------------------
# K-Means Clustering Visualization using Custom KMeans
# ---------------------------------------------------

# (Optional) Used earlier for synthetic data generation
# from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt
import pandas as pd
from kmeans import KMeans

# #centroids = [(-5,-5),(5,5),(-2.5,2.5),(2.5,-2.5)]
# #cluster_std = [1,1,1,1]
#
# #X,y = make_blobs(n_samples=100,cluster_std=cluster_std,centers=centroids,n_features=2,random_state=2)
#
# #plt.scatter(X[:,0],X[:,1])

# ---------------------------------------------------
# STEP 1: Load dataset from CSV file
# ---------------------------------------------------
df = pd.read_csv('student_clustering.csv')

# Convert DataFrame to NumPy array
X = df.iloc[:, :].values


# ---------------------------------------------------
# STEP 2: Initialize and train K-Means model
# ---------------------------------------------------
km = KMeans(n_clusters=4, max_iter=500)

# Fit the model and get cluster labels
y_means = km.fit_predict(X)


# ---------------------------------------------------
# STEP 3: Visualize clusters (only works for 2 features)
# ---------------------------------------------------
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1],
            color='red', label='Cluster 0')

plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1],
            color='blue', label='Cluster 1')

plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1],
            color='green', label='Cluster 2')

plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1],
            color='yellow', label='Cluster 3')

# Plot centroids
plt.scatter(km.centroids[:, 0], km.centroids[:, 1],
            color='black', marker='X', s=200, label='Centroids')

plt.title("Student Clustering using K-Means (From Scratch)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
