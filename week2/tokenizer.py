import json
import re

class BPETokenizer:
    def __init__(self, vocab_file, merges_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_file, "r", encoding="utf-8") as f:
            merges = f.read().splitlines()[1:]
        self.bpe_ranks = dict(zip([tuple(merge.split()) for merge in merges], range(len(merges))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.cache = {}

    def get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        text = re.findall(r"\S+", text)
        tokens = []
        for word in text:
            word = list(word)
            word[-1] += '</w>'
            bpe_tokens = self.bpe(word).split()
            tokens.extend([self.encoder[token] for token in bpe_tokens])
        return tokens

    def decode(self, tokens):
        words = [self.decoder[token] for token in tokens]
        text = ''.join(words).replace('</w>', ' ')
        return text.strip()
