from sklearn.datasets import make_moons
from random import shuffle

def load_data():
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    data = list(zip(X, y))
    shuffle(data)
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]
    X_train, y_train = zip(*train)
    X_test, y_test = zip(*test)
    return list(X_train), list(y_train), list(X_test), list(y_test)
