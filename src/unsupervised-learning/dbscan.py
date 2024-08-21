import numpy as np

class DBSCANScratch:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def compute_distance(self, point1, point2):
        if self.metric == 'euclidean':
            return np.linalg.norm(point1 - point2)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        elif self.metric == 'minkowski':
            p = 3 
            return np.sum(np.abs(point1 - point2) ** p) ** (1/p)
        else:
            raise ValueError(f"Metrik {self.metric} tidak diketahui")

    def region_query(self, X, index):
        neighbors = []
        for i in range(X.shape[0]):
            if self.compute_distance(X[index], X[i]) < self.eps:
                neighbors.append(i)
        return np.array(neighbors)

    def expand_cluster(self, X, labels, index, neighbors, cluster_id):
        labels[index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if labels[neighbor] == -1:  
                labels[neighbor] = cluster_id
            elif labels[neighbor] == 0:  
                labels[neighbor] = cluster_id
                new_neighbors = self.region_query(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            i += 1

    def fit(self, X):
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0
        
        for i in range(n_samples):
            if labels[i] != 0:  # Already processed
                continue
            
            neighbors = self.region_query(X, i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1  # Mark as noise
            else:
                cluster_id += 1
                self.expand_cluster(X, labels, i, neighbors, cluster_id)
        
        self.labels_ = labels
        return labels

    def fit_predict(self, X):
        return self.fit(X)
