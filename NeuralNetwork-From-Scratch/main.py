import numpy as np
from network import Network
from fully_connected import FullyConnectedLayer
from activation import ActivationReLU

# Données d'entraînement pour résoudre le problème XOR
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Initialisation du réseau de neurones
network = Network()
network.add_layer(FullyConnectedLayer(2, 4))  # Couche fully connected avec 2 entrées et 4 sorties
network.add_layer(ActivationReLU())           # Couche d'activation ReLU
network.add_layer(FullyConnectedLayer(4, 1))  # Couche fully connected avec 4 entrées et 1 sortie

# Fonction pour tester les prédictions du réseau
def test_network(network, X):
    print("Prédictions du réseau :")
    predictions = network.predict(X)
    for i, prediction in enumerate(predictions):
        print(f"Entrée: {X[i]} -> Prédiction: {prediction}")

# Tester le réseau avant l'entraînement
print("Avant l'entraînement :")
test_network(network, X_train)

# Entraîner le réseau sur 1000 époques avec un learning rate de 0.1
network.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Tester le réseau après l'entraînement
print("\nAprès l'entraînement :")
test_network(network, X_train)
