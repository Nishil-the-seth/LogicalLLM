import numpy as np

class CausalSelfAttention:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_k = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_v = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_o = np.random.randn(embed_dim, embed_dim) * 0.01

    def split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def merge_heads(self, x):
        b, h, t, d = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, h * d)

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / np.sum(ex, axis=-1, keepdims=True)

    def causal_mask(self, size):
        return np.tril(np.ones((size, size)))

    def __call__(self, x):
        b, t, _ = x.shape
        q = x @ self.W_q
        k = x @ self.W_k
        v = x @ self.W_v

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        mask = self.causal_mask(t)
        scores = scores * mask[:, None, None, :] - 1e10 * (1 - mask[:, None, None, :])

        attn = self.softmax(scores)
        out = attn @ v
        out = self.merge_heads(out)
        return out @ self.W_o
