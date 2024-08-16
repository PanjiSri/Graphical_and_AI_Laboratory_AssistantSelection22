import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.output = None
        print(f"Layer initialized: input_size={input_size}, output_size={output_size}, activation={activation}")

    def forward(self, input):
        self.input = input
        self.z = np.dot(input, self.weights) + self.biases
        self.output = self.apply_activation(self.z)
        print(f"Forward pass: input shape={input.shape}, output shape={self.output.shape}")
        return self.output

    def backward(self, output_gradient, learning_rate):
        print(f"Backward pass: output_gradient shape={output_gradient.shape}, weights shape={self.weights.shape}")
        if self.activation == 'softmax':
            input_gradient = np.dot(output_gradient, self.weights.T)
            weights_gradient = np.dot(self.input.T, output_gradient)
        else:
            activation_gradient = self.apply_activation_derivative(self.z) * output_gradient
            input_gradient = np.dot(activation_gradient, self.weights.T)
            weights_gradient = np.dot(self.input.T, activation_gradient)
        
        biases_gradient = np.sum(activation_gradient if self.activation != 'softmax' else output_gradient, axis=0, keepdims=True)
        
        print(f"Gradients: weights={weights_gradient.shape}, biases={biases_gradient.shape}, input={input_gradient.shape}")
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
            raise ValueError(f"Unsupported activation function: {self.activation}")

    def apply_activation_derivative(self, x):
        if self.activation == 'sigmoid':
            s = self.apply_activation(x)
            return s * (1 - s)
        elif self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'linear':
            return np.ones_like(x)
        elif self.activation == 'softmax':
            # Softmax derivative is handled in backward method
            return x
        else:
            raise ValueError(f"Unsupported activation function: {self.activation}")

class ANN_Scratch:
    def __init__(self, layer_sizes, activations, loss='mse', regularization=None, reg_lambda=0.01, learning_rate=0.01, epochs=1000, batch_size=32):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))
        self.loss = loss
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        print(f"ANN initialized with {len(self.layers)} layers")

    def forward(self, X):
        for i, layer in enumerate(self.layers):
            X = layer.forward(X)
            print(f"Layer {i+1} output shape: {X.shape}")
        return X

    def backward(self, X, y, learning_rate):
        output = self.forward(X)
        print(f"Final output shape: {output.shape}")
        gradient = self.loss_derivative(y, output)
        print(f"Initial gradient shape: {gradient.shape}")
        
        for i, layer in enumerate(reversed(self.layers)):
            gradient = layer.backward(gradient, learning_rate)
            print(f"Layer {len(self.layers)-i} gradient shape: {gradient.shape}")

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]
                self.backward(X_batch, y_batch, self.learning_rate)
            
            if epoch % 100 == 0:
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
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
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
            raise ValueError(f"Unsupported loss function: {self.loss}")
