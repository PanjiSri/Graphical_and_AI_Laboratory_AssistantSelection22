import numpy as np

class PCA_Scratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        X_centered = X - np.mean(X, axis=0)

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.components_)
    
    def explained_variance_ratio_(self):
        return self.explained_variance_