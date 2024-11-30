import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        # Initialisation aléatoire des poids et des biais
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        # Produit matriciel des entrées avec les poids + biais
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, loss_gradient, learning_rate):
        # Calcul des gradients pour les poids et biais
        weights_gradient = np.dot(self.inputs.T, loss_gradient)
        biases_gradient = np.sum(loss_gradient, axis=0, keepdims=True)

        # Mise à jour des poids et biais
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        # Retourner le gradient pour la couche précédente
        return np.dot(loss_gradient, self.weights.T)
