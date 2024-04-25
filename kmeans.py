kmeans 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

df.to_csv('data.csv', index=False)

df = pd.read_csv('data.csv')

kmeans = KMeans(n_clusters=4)
kmeans.fit(df)

plt.figure(figsize=(12, 6))

plt.scatter(df['Feature1'], df['Feature2'], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.5)

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')

plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()