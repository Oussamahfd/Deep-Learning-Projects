import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
    
    def forward(self, X):
        self.input = X
        self.output = np.dot(X, self.weights) + self.biases
        return self.output
    
    def backward(self, d_output, learning_rate):
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_biases = np.sum(d_output, axis=0, keepdims=True)
        
        # Mise à jour des poids et biais
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input
