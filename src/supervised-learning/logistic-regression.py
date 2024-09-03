import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.01, loss_function='cross_entropy'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.loss_function = loss_function
        
    def sigmoid(self, z):
        z = np.clip(z, -250, 250)  
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y, y_predict, m):
        epsilon = 1e-15 
        if self.loss_function == 'cross_entropy':
            cost = -1/m * np.sum(y * np.log(y_predict + epsilon) + (1 - y) * np.log(1 - y_predict + epsilon))
        elif self.loss_function == 'squared_error':
            cost = np.mean((y_predict - y) ** 2)

        if self.regularization == 'l2':
            cost += (self.lambda_ / (2 * m)) * np.sum(np.square(self.w))
        elif self.regularization == 'l1':
            cost += (self.lambda_ / m) * np.sum(np.abs(self.w))
        
        return cost
    
    def fit(self, X_train, y_train, method='gradient_descent'):
        m, n = X_train.shape
        self.w = np.zeros(n)
        self.b = 0
        self.cost_history = []

        # print(f"Debug: Memulai pelatihan dengan metode {method}")
        # print(f"Debug: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        for i in range(self.num_iterations):
            z = np.dot(X_train, self.w) + self.b
            y_predict = self.sigmoid(z)
            
            if np.any(np.isnan(y_predict)):
                print(f"Error: Prediksi sigmoid menghasilkan NaN pada iterasi {i+1}")
                return
            
            gradient = 1/m * np.dot(X_train.T, (y_predict - y_train))
            db = 1/m * np.sum(y_predict - y_train)
            
            # if i % 100 == 0 or i == 0:
            #     print(f"Debug Iterasi {i+1}: Cost = {self.compute_cost(y_train, y_predict, m):.4f}, Gradient mean = {np.mean(gradient)}, db = {db}")
            
            if np.any((y_predict < 1e-10) | (y_predict > 1 - 1e-10)):
                print(f"Warning: Prediksi sigmoid mendekati 0 atau 1 pada iterasi {i+1}. Ini bisa menyebabkan Hessian ill-conditioned.")
            
            if method == 'gradient_descent':
                if self.regularization == 'l2':
                    gradient += (self.lambda_ / m) * self.w
                elif self.regularization == 'l1':
                    gradient += (self.lambda_ / m) * np.sign(self.w)
                
                dw = gradient
                
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            elif method == 'newton':
                if self.regularization == 'l1':
                    raise ValueError("Metode Newton tidak kompatibel dengan regularisasi L1. Gunakan metode gradient descent atau pilih regularisasi L2.")
                
                # Kalkulasi Hessian dengan stabilisasi
                D = np.diagflat(y_predict * (1 - y_predict))
                
                # if np.any(D.diagonal() < 1e-10):
                #     print(f"Warning: Nilai diagonal D sangat kecil pada iterasi {i+1}. Ini bisa menyebabkan Hessian singular.")
                
                try:
                    hessian = 1/m * np.dot(X_train.T, D @ X_train)
                    if self.regularization == 'l2':
                        hessian += (self.lambda_ / m) * np.eye(n)
                    
                    hessian += 1e-5 * np.eye(n)
                    
                    hessian_inv = np.linalg.inv(hessian)

                except np.linalg.LinAlgError:
                    print(f"Error: Hessian matrix singular pada iterasi ke {i+1}. Menggunakan pseudo-inverse.")
                    hessian_inv = np.linalg.pinv(hessian)
                
                update_w = np.dot(hessian_inv, gradient)
                
                self.w -= update_w
                self.b -= self.learning_rate * db  
            
            else:
                print(f"Error: Metode tidak dikenal '{method}'")
                return

            cost = self.compute_cost(y_train, y_predict, m)
            self.cost_history.append(cost)
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Iterasi {i+1}/{self.num_iterations}: Biaya = {cost:.4f}")
        
    def predict(self, X_test):
        z = np.dot(X_test, self.w) + self.b
        return self.sigmoid(z) >= 0.5

    def predict_proba(self, X_test):
        z = np.dot(X_test, self.w) + self.b
        return self.sigmoid(z)
