import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation, init_method='he'):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        if init_method == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        elif init_method == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1. / input_size)
        else:
            raise ValueError("Metode inisialisasi tidak dikenal. Gunakan 'he' atau 'xavier'.")

        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.apply_activation(self.z)
        return self.output

    def backward(self, output_gradient, learning_rate, batch_size, regularization=None, reg_lambda=0.01):
        if self.activation == 'softmax':
            activation_gradient = output_gradient
        else:
            activation_gradient = self.apply_activation_derivative(self.z) * output_gradient
        
        input_gradient = np.dot(activation_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, activation_gradient) / batch_size
        biases_gradient = np.sum(activation_gradient, axis=0, keepdims=True) / batch_size

        # Menambahkan regularisasi pada weights_gradient
        if regularization == 'l1':
            weights_gradient += reg_lambda * np.sign(self.weights) / batch_size
        elif regularization == 'l2':
            weights_gradient += reg_lambda * self.weights / batch_size

        # Update weights dan biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient
        
        return input_gradient

    def apply_activation(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -709, 709)))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'linear':
            return x
        elif self.activation == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        else:
            raise ValueError(f"Fungsi aktivasi tidak didukung: {self.activation}")

    def apply_activation_derivative(self, x):
        if self.activation == 'sigmoid':
            s = self.apply_activation(x)
            return s * (1 - s)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'linear':
            return np.ones_like(x)
        else:
            raise ValueError(f"Fungsi aktivasi tidak didukung: {self.activation}")

class ANN_Scratch:
    def __init__(self, layer_sizes, activations, loss='mse', regularization=None, reg_lambda=0.01, learning_rate=0.01, epochs=1000, batch_size=32, init_method='he'):
        self.layers = []
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.loss = loss
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i], init_method)
            self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, X, y):
        output = self.forward(X)
        gradient = self.loss_derivative(y, output)
        
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, self.learning_rate, self.batch_size, self.regularization, self.reg_lambda)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                self.backward(X_batch, y_batch)
            
            # if epoch % 10 == 0: 
            loss = self.calculate_loss(X, y)
            print(f"Epoch {epoch}, Loss: {loss}")  

    def predict(self, X):
        return self.forward(X)

    def calculate_loss(self, X, y):
        output = self.forward(X)
        if self.loss == 'mse':
            loss = np.mean((y - output) ** 2)
        elif self.loss == 'binary_crossentropy':
            output = np.clip(output, 1e-15, 1 - 1e-15)
            loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        else:
            raise ValueError(f"Fungsi loss tidak didukung: {self.loss}")
        
        if self.regularization == 'l1':
            l1_loss = self.reg_lambda * sum(np.sum(np.abs(layer.weights)) for layer in self.layers)
            loss += l1_loss
        elif self.regularization == 'l2':
            l2_loss = self.reg_lambda * sum(np.sum(layer.weights ** 2) for layer in self.layers)
            loss += l2_loss
        
        return loss

    def loss_derivative(self, y, output):
        if self.loss == 'mse':
            return 2 * (output - y) / y.size
        elif self.loss == 'binary_crossentropy':
            output = np.clip(output, 1e-15, 1 - 1e-15)
            return (output - y) / (output * (1 - output))
        else:
            raise ValueError(f"Fungsi loss tidak didukung: {self.loss}")
