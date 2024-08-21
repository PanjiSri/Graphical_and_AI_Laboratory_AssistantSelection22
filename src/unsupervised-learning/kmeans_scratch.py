import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=3, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def initialize_centroids(self, X):
        np.random.seed(42)
        random_indices = np.random.permutation(X.shape[0])
        centroids = X[random_indices[:self.n_clusters]]
        return centroids

    def compute_distances(self, X, centroids):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
        return distances

    def assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            centroids[i] = X[labels == i].mean(axis=0)
        return centroids

    def fit(self, X):
        self.centroids = self.initialize_centroids(X)

        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            distances = self.compute_distances(X, old_centroids)
            labels = self.assign_clusters(distances)
            self.centroids = self.compute_centroids(X, labels)

            if np.all(old_centroids == self.centroids):
                break

        return labels

    def predict(self, X):
        distances = self.compute_distances(X, self.centroids)
        return self.assign_clusters(distances)
