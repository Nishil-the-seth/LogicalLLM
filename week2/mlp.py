import numpy as np

class FeedForward:
    def __init__(self, embed_dim, hidden_dim):
        self.W1 = np.random.randn(embed_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embed_dim) * 0.01
        self.b2 = np.zeros(embed_dim)

    def __call__(self, x):
        x = x @ self.W1 + self.b1
        x = np.maximum(0, x)
        x = x @ self.W2 + self.b2
        return x
