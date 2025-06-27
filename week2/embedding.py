import numpy as np

class Embedder:
    def __init__(self, vocab_size, embedding_dim, max_len=128):
        self.embedding_dim = embedding_dim
        self.word_embeddings = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.position_embeddings = np.random.randn(max_len, embedding_dim) * 0.01

    def __call__(self, token_ids):
        seq_len = len(token_ids)
        token_embeds = [self.word_embeddings[token_id] for token_id in token_ids]
        pos_embeds = [self.position_embeddings[i] for i in range(seq_len)]
        return np.array(token_embeds) + np.array(pos_embeds)
