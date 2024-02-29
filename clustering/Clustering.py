import numpy as np
import sklearn.datasets as ds
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=3, max_iterations=100, tolerance=1e-4):
        self.k = k 
        self.max_iterations = max_iterations 
        self.tolerance = tolerance
        self.centroids = [] # centroid locations
        self.clusters = [] # clusters of all data

    def fit(self, X):
        # Randomly init centroids
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            self.assign_clusters(X)
            self.update_centroids(X)

            # Check for convergence
            if np.sum((self.centroids - old_centroids) / old_centroids * 100.0) < self.tolerance:
                break

    def euclidean_distance(self, point):
        return np.sqrt(np.sum((point - self.centroids)**2, axis = 1))

    # Calc smallest distance for each point, assign cluster
    def assign_clusters(self, X):
        clusters = []
        for point in X:
            distances = self.euclidean_distance(point)
            closest_centroid = np.argmin(distances)
            clusters.append(closest_centroid)
        self.clusters = np.array(clusters) 
    
    # Calculate mean of all points with centroid, reset centroid to that mean
    def update_centroids(self, X):
        centroids = np.zeros((self.k, X.shape[1]))
        for cluster in range(self.k):
            cluster_points = X[np.array(self.clusters) == cluster]
            if len(cluster_points) > 0:
                centroids[cluster] = np.mean(cluster_points, axis=0)
        return centroids

    # Calc smallest distance for each point --> prediction
    def predict(self, X):
        predictions = []
        for point in X:
            distances = self.euclidean_distance(point)
            closest_centroid = np.argmin(distances)
            predictions.append(closest_centroid)
        return predictions
    
def plot_clusters(X, kmeans):
    plt.figure(figsize=(8, 6))
    
    # Plot data points
    clusters = kmeans.clusters
    for i in range(kmeans.k):
        cluster_points = X[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}')
    
    # Plot centroids
    centroids = kmeans.centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', alpha=0.5)
    
    plt.title('KMeans Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    X, _ = ds.make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
    
    kmeans = KMeans(k=4, max_iterations=100)
    kmeans.fit(X)
    
    predictions = kmeans.predict(X)

    print(X)
    print(predictions)

    plot_clusters(X, kmeans)

main()