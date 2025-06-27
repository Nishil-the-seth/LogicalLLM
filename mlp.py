import random
import math

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]

        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.b2 = random.uniform(-1, 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, x):
        self.z1 = []
        self.a1 = []
        for i in range(self.hidden_size):
            z = sum([self.w1[i][j] * x[j] for j in range(self.input_size)]) + self.b1[i]
            self.z1.append(z)
            self.a1.append(self.sigmoid(z))
        self.z2 = sum([self.w2[i] * self.a1[i] for i in range(self.hidden_size)]) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, x, y, learning_rate):
        y_pred = self.a2
        d_a2 = (y_pred - y) * self.sigmoid_derivative(self.z2)
        for i in range(self.hidden_size):
            self.w2[i] -= learning_rate * d_a2 * self.a1[i]
        self.b2 -= learning_rate * d_a2
        d_hidden = [self.w2[i] * d_a2 * self.sigmoid_derivative(self.z1[i]) for i in range(self.hidden_size)]
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.w1[i][j] -= learning_rate * d_hidden[i] * x[j]
            self.b1[i] -= learning_rate * d_hidden[i]
        return - (y * math.log(y_pred + 1e-8) + (1 - y) * math.log(1 - y_pred + 1e-8))

    def train_batch(self, batch_x, batch_y, learning_rate):
        total_loss = 0
        for x, y in zip(batch_x, batch_y):
            self.forward(x)
            total_loss += self.backward(x, y, learning_rate)
        return total_loss / len(batch_x)

    def predict(self, x):
        return 1 if self.forward(x) >= 0.5 else 0
