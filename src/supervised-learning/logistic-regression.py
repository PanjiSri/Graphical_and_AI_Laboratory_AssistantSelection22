import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, num_iterations=1000, regularization=None, lambda_=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        
    def sigmoid(self, z):
        z = np.array(z)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y, y_predict, m):
        cost = -1/m * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))
        
        if self.regularization == 'l2':
            cost += (self.lambda_ / (2 * m)) * np.sum(np.square(self.w)) 
        elif self.regularization == 'l1':
            cost += (self.lambda_ / m) * np.sum(np.abs(self.w))  
        
        return cost
    
    def fit(self, X_train, y_train):
        m, n = X_train.shape
        self.w = np.zeros(n)
        self.b = 0
        self.cost_history = []
        
        for i in range(self.num_iterations):
            z = np.dot(X_train, self.w) + self.b
            y_predict = self.sigmoid(z)
            
            dw = 1/m * np.dot(X_train.T, (y_predict - y_train))
            db = 1/m * np.sum(y_predict - y_train)
            
            if self.regularization == 'l2':
                dw += (self.lambda_ / m) * self.w
            elif self.regularization == 'l1':
                dw += (self.lambda_ / m) * np.sign(self.w)
                
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            cost = self.compute_cost(y_train, y_predict, m)
            self.cost_history.append(cost)
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Iteration {i+1}/{self.num_iterations}: Cost = {cost:.4f}")
        
    def predict(self, X_test):
        z = np.dot(X_test, self.w) + self.b
        return self.sigmoid(z) >= 0.5
