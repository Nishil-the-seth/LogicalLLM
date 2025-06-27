from tokenizer import BPETokenizer
from model import Transformer
import numpy as np

vocab_file = "vocab.json"
merges_file = "merges.txt"
tokenizer = BPETokenizer(vocab_file, merges_file)

text = "hello world"
tokens = tokenizer.encode(text)

model = Transformer(vocab_size=50257, embed_dim=64, max_len=128, num_heads=4, ff_dim=256)
output = model(tokens)

print("Input Text:", text)
print("Token IDs:", tokens)
print("Output Shape:", output.shape)
print("Output Vector (first token):", output[0])
