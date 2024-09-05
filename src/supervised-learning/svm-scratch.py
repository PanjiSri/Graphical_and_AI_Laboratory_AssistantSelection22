import numpy as np

class SVM_Scratch:
    def __init__(self, learning_rate=0.00001, num_iterations=1000, lambda_param=0.01, kernel='linear', degree=3, gamma=0.1):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.w = None
        self.b = None
        self.X_train = None

    def linear_kernel(self, X, X2):
        return np.dot(X, X2.T)
    
    def polynomial_kernel(self, X, X2):
        return np.power((1 + np.dot(X, X2.T)), self.degree)
    
    def rbf_kernel(self, X, X2):
        if self.gamma is None:
            self.gamma = 1 / X.shape[1]
        K = np.exp(-self.gamma * np.linalg.norm(X[:, np.newaxis] - X2, axis=2) ** 2)
        return K

    def compute_kernel(self, X, X2):
        if self.kernel == 'linear':
            return self.linear_kernel(X, X2)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(X, X2)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(X, X2)
        else:
            raise ValueError("Kernel tidak diketahui")

    def fit(self, X_train, y_train):
        self.X_train = X_train 
        y_train = np.where(y_train <= 0, -1, 1)
        m, n = X_train.shape
        self.w = np.zeros(m) 
        self.b = 0

        # print(f"[DEBUG] Mulai fit: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        # print(f"[DEBUG] Kernel yang dipilih: {self.kernel}")

        # Menggunakan kernel pada pelatihan
        K = self.compute_kernel(X_train, X_train)
        # print(f"[DEBUG] Kernel matrix K.shape={K.shape}")

        for i in range(self.num_iterations):
            for idx in range(m):
                condition = y_train[idx] * (np.dot(K[idx], self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(K[idx], y_train[idx]))
                    self.b -= self.learning_rate * y_train[idx]
            
            if i % 50 == 0:
                loss = self._calculate_loss(K, y_train)
                print(f"[DEBUG] Iterasi {i}: Loss = {loss}, w.shape={self.w.shape}, b={self.b}")
    
    def _calculate_loss(self, K, y):
        hinge_loss = np.maximum(0, 1 - y * (np.dot(K, self.w) - self.b))
        loss = 0.5 * np.dot(self.w, self.w) + self.lambda_param * np.sum(hinge_loss)
        # print(f"[DEBUG] Calculated loss: {loss}")
        return loss

    def predict(self, X_test):
        # Menggunakan kernel antara X_test dan self.X_train
        K = self.compute_kernel(X_test, self.X_train)
        # print(f"[DEBUG] Predict: X_test.shape={X_test.shape}, Kernel K.shape={K.shape}")

        approx = np.dot(K, self.w) - self.b
        # print(f"[DEBUG] Predict: approx.shape={approx.shape}, w.shape={self.w.shape}, b={self.b}")

        return np.where(np.sign(approx) == -1, 0, 1)
