from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Example data: 2D points
X = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10], [10, 11]])

# Create and fit k-means model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Get cluster assignments and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster Labels:", labels)
print("Centroids:", centroids)

# Visualize
for i in range(2):  # Two clusters
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', label='Centroids')
plt.legend()
plt.show()