import numpy as np

class LADOP:
    def __init__(self):
        self.mean_pattern = None
      
    def fit(self, X):
        self.mean_pattern = np.mean(
            X,
            axis=0
        )

    def score(self, X):
        scores = np.mean(
            (X - self.mean_pattern) ** 2,
            axis=(1, 2)
        )

        return scores
