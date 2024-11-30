import numpy as np
from fully_connected import FullyConnectedLayer  # Ajouter cet import pour FullyConnectedLayer

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, loss_gradient, learning_rate):
        for layer in reversed(self.layers):
            if isinstance(layer, FullyConnectedLayer):  # Vérifie si la couche est une FullyConnectedLayer
                loss_gradient = layer.backward(loss_gradient, learning_rate)
            else:
                loss_gradient = layer.backward(loss_gradient)

    def train(self, X_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            predictions = self.forward(X_train)
            loss = self.compute_loss(predictions, y_train)
            loss_gradient = self.compute_loss_gradient(predictions, y_train)
            self.backward(loss_gradient, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs}, Perte: {loss}")

    def compute_loss(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)

    def compute_loss_gradient(self, predictions, targets):
        return 2 * (predictions - targets) / targets.size

    def predict(self, inputs):
        return self.forward(inputs)
