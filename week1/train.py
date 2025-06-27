from mlp import MLP
from dataset import load_data
from plot import plot_loss
import random

X_train, y_train, X_test, y_test = load_data()
model = MLP(input_size=2, hidden_size=4, output_size=1)

epochs = 1000
batch_size = 16
learning_rate = 0.1
losses = []

for epoch in range(epochs):
    combined = list(zip(X_train, y_train))
    random.shuffle(combined)
    X_train[:], y_train[:] = zip(*combined)

    total_loss = 0
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        loss = model.train_batch(batch_x, batch_y, learning_rate)
        total_loss += loss
    avg_loss = total_loss / (len(X_train) // batch_size)
    losses.append(avg_loss)

plot_loss(losses)
