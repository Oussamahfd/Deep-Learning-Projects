import numpy as np

class ActivationReLU:
    def forward(self, inputs):
        self.inputs = inputs
        # Fonction d'activation ReLU: max(0, input)
        return np.maximum(0, inputs)

    def backward(self, loss_gradient):
        # Dérivée de ReLU : 1 si inputs > 0, sinon 0
        return loss_gradient * (self.inputs > 0)

