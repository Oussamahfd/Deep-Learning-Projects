import numpy as np

class LossMeanSquaredError:
    def forward(self, predictions, targets):
        # Calcul de l'erreur quadratique moyenne (MSE)
        self.predictions = predictions
        self.targets = targets
        loss = np.mean((predictions - targets) ** 2)
        return loss
    
    def backward(self):
        # Calcul du gradient de la perte
        d_output = 2 * (self.predictions - self.targets) / self.targets.size
        return d_output
