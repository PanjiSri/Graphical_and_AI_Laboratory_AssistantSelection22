import numpy as np

class SVM_Scratch:
    def __init__(self, learning_rate=0.001, num_iterations=10, lambda_param=0.01, kernel='linear', degree=3):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.lambda_param = lambda_param
        self.kernel = kernel
        self.degree = degree
        self.w = None
        self.b = None

    def linear_kernel(self, X):
        return X
    
    def polynomial_kernel(self, X):
        return np.power((1 + np.dot(X, X.T)), self.degree)

    def compute_kernel(self, X):
        if self.kernel == 'linear':
            return self.linear_kernel(X)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(X)
        else:
            raise ValueError("Kernel tidak diketahui")

    def fit(self, X_train, y_train):
        y_train = np.where(y_train <= 0, -1, 1)
        m, n = X_train.shape
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.num_iterations):
            for idx, x_i in enumerate(X_train):
                condition = y_train[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_train[idx]))
                    self.b -= self.learning_rate * y_train[idx]
    
    def predict(self, X_test):
        approx = np.dot(X_test, self.w) - self.b
        return np.where(np.sign(approx) == -1, 0, 1)  