from dataset import load_data
from mlp import MLP
import matplotlib.pyplot as plt

# Load and split data
X_train, y_train, X_test, y_test = load_data()

# Create model
model = MLP(input_size=2, hidden_size=4, output_size=1)

# Train
epochs = 1000
batch_size = 16
losses = []

for epoch in range(epochs):
    # Manually split into mini-batches and call model.train_batch(...)
    # Append loss to losses[]
    pass

# Plot loss
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
