"""


Architecture: [2] -> [4] -> [4] -> [1]
Task:         Learn XOR  (the classic impossible-for-perceptron problem)
Training:     Backpropagation + gradient descent
"""

import math
import random
import os
import time

def sigmoid(x):
    """Squishes any number into (0, 1)."""
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))

def sigmoid_derivative(x):
    """Gradient of sigmoid — used in backprop."""
    s = sigmoid(x)
    return s * (1.0 - s)

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def tanh(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1.0 - math.tanh(x) ** 2

def mse_loss(predictions, targets):
    """Mean Squared Error loss."""
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

def zeros(rows, cols):
    return [[0.0] * cols for _ in range(rows)]

def rand_matrix(rows, cols, scale=1.0):
    """Xavier-ish initialization."""
    limit = scale * math.sqrt(6.0 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def dot(mat, vec):
    """Matrix × vector → vector."""
    return [sum(mat[i][j] * vec[j] for j in range(len(vec))) for i in range(len(mat))]

def mat_transpose(mat):
    rows, cols = len(mat), len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


class Layer:
    def __init__(self, in_size, out_size, activation='sigmoid'):
        self.W = rand_matrix(out_size, in_size)
        self.b = [0.0] * out_size
        self.activation = activation

       
        self.last_input   = []
        self.last_z       = []
        self.last_output  = []

      
        self.dW = zeros(out_size, in_size)
        self.db = [0.0] * out_size

    def _activate(self, z):
        if self.activation == 'sigmoid': return sigmoid(z)
        if self.activation == 'relu':    return relu(z)
        if self.activation == 'tanh':    return tanh(z)
        return z  # linear

    def _activate_prime(self, z):
        if self.activation == 'sigmoid': return sigmoid_derivative(z)
        if self.activation == 'relu':    return relu_derivative(z)
        if self.activation == 'tanh':    return tanh_derivative(z)
        return 1.0

    def forward(self, x):
        self.last_input = x[:]
        self.last_z     = [sum(self.W[i][j] * x[j] for j in range(len(x))) + self.b[i]
                           for i in range(len(self.b))]
        self.last_output = [self._activate(z) for z in self.last_z]
        return self.last_output

    def backward(self, delta_next):
      
       
        delta = [delta_next[i] * self._activate_prime(self.last_z[i])
                 for i in range(len(self.last_z))]

        
        for i in range(len(self.b)):
            self.db[i] += delta[i]
            for j in range(len(self.last_input)):
                self.dW[i][j] += delta[i] * self.last_input[j]

     
        WT = mat_transpose(self.W)
        return dot(WT, delta)

    def update(self, lr, batch_size):
        """Apply accumulated gradients, then zero them."""
        for i in range(len(self.b)):
            self.b[i] -= lr * self.db[i] / batch_size
            for j in range(len(self.W[i])):
                self.W[i][j] -= lr * self.dW[i][j] / batch_size
       
        self.dW = zeros(len(self.b), len(self.W[0]))
        self.db = [0.0] * len(self.b)

class NeuralNetwork:
    def __init__(self, layer_sizes, activations=None):
        """
        layer_sizes: e.g. [2, 4, 4, 1]
        activations: list of strings, length = len(layer_sizes) - 1
        """
        if activations is None:
            activations = ['tanh'] * (len(layer_sizes) - 2) + ['sigmoid']

        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            for i in range(len(layer_sizes) - 1)
        ]
        self.loss_history = []

    def predict(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _compute_output_delta(self, output, target):
        """MSE gradient at output layer (before activation derivative)."""
        return [2.0 * (output[i] - target[i]) / len(output) for i in range(len(output))]

    def train_step(self, X, Y, lr=0.1):
        """
        X: list of input vectors
        Y: list of target vectors
        """
        total_loss = 0.0

        for x, y in zip(X, Y):
            # ── forward ──
            output = self.predict(x)
            total_loss += mse_loss(output, y)

 
            delta = self._compute_output_delta(output, y)

            for layer in reversed(self.layers):
                delta = layer.backward(delta)

      
        for layer in self.layers:
            layer.update(lr, len(X))

        avg_loss = total_loss / len(X)
        self.loss_history.append(avg_loss)
        return avg_loss

    def train(self, X, Y, epochs=5000, lr=0.1, verbose=True, log_every=500):
        print(f"\n  Training {len(self.layers)}-layer network")
        print(f"  Samples: {len(X)}  |  LR: {lr}  |  Epochs: {epochs}\n")

        start = time.time()
        for epoch in range(1, epochs + 1):
            loss = self.train_step(X, Y, lr)

            if verbose and epoch % log_every == 0:
                bar = self._loss_bar(loss)
                elapsed = time.time() - start
                print(f"  epoch {epoch:>5} │ loss {loss:.6f} │ {bar} │ {elapsed:.1f}s")

        print(f"\n  Done. Final loss: {self.loss_history[-1]:.6f}")
        return self

    def _loss_bar(self, loss, width=20):
        filled = int((1.0 - min(1.0, loss * 10)) * width)
        return "█" * filled + "░" * (width - filled)

    def summary(self):
        print("\n  ┌─ Network Architecture ─────────────────────┐")
        for i, layer in enumerate(self.layers):
            rows = len(layer.W)
            cols = len(layer.W[0])
            params = rows * cols + rows
            print(f"  │  Layer {i+1}: {cols:>2} → {rows:>2}  "
                  f"({layer.activation:<8})  params: {params}")
        total = sum(len(l.W) * len(l.W[0]) + len(l.b) for l in self.layers)
        print(f"  │  Total parameters: {total}")
        print("  └────────────────────────────────────────────┘\n")

def run_xor():
    print("=" * 56)
    print("  XOR Problem — impossible for linear models")
    print("=" * 56)

    # XOR truth table
    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0],       [1.0],       [1.0],       [0.0]]

    net = NeuralNetwork([2, 8, 8, 1], activations=['tanh', 'tanh', 'sigmoid'])
    net.summary()
    net.train(X, Y, epochs=8000, lr=0.05, log_every=1000)

    print("\n  ┌─ Predictions ──────────────────────────────────┐")
    print("  │   Input     │  Target  │  Output  │  Correct   │")
    print("  ├─────────────┼──────────┼──────────┼────────────┤")
    correct = 0
    for x, y in zip(X, Y):
        pred = net.predict(x)[0]
        rounded = round(pred)
        ok = "✓" if rounded == int(y[0]) else "✗"
        if rounded == int(y[0]): correct += 1
        print(f"  │  {x[0]:.0f} XOR {x[1]:.0f}   │   {y[0]:.1f}    │  {pred:.4f}  │    {ok}       │")
    print("  └─────────────┴──────────┴──────────┴────────────┘")
    print(f"\n  Accuracy: {correct}/{len(X)}  ({100*correct//len(X)}%)")


def print_loss_curve(history, width=50, height=12):
    if not history:
        return
    sample = history[::max(1, len(history)//width)][:width]
    mn, mx = min(sample), max(sample)
    rng = mx - mn or 1e-9

    print("\n  ┌─ Training loss curve " + "─" * (width - 2) + "┐")
    for row in range(height, -1, -1):
        thresh = mn + (row / height) * rng
        line = "".join("█" if v >= thresh else " " for v in sample)
        label = f"{thresh:.4f}" if row % (height // 3) == 0 else "      "
        print(f"  │ {label} {line}│")
    print("  └" + "─" * (width + 8) + "┘")
    print(f"  {'epoch 0':>{width//2+8}}{'epoch ' + str(len(history)):>{width//2}}")

def print_decision_boundary(net, cols=40, rows=20):
    print("\n  ┌─ Decision boundary (XOR) " + "─" * (cols - 4) + "┐")
    for row in range(rows):
        y_val = 1.0 - row / (rows - 1)
        line = ""
        for col in range(cols):
            x_val = col / (cols - 1)
            pred = net.predict([x_val, y_val])[0]
            if pred > 0.75:    line += "██"
            elif pred > 0.5:   line += "▓▓"
            elif pred > 0.25:  line += "░░"
            else:              line += "  "
        print(f"  │{line}│")
    print("  └" + "─" * (cols * 2) + "┘")
    print("  (dark = predicts 1, light = predicts 0)\n")
─

if __name__ == "__main__":
    random.seed(42)

    net = NeuralNetwork([2, 8, 8, 1], activations=['tanh', 'tanh', 'sigmoid'])
    net.summary()

    X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    Y = [[0.0],       [1.0],       [1.0],       [0.0]]

    net.train(X, Y, epochs=8000, lr=0.05, log_every=1000)

    print_loss_curve(net.loss_history)
    print_decision_boundary(net)

    print("\n  ┌─ Final predictions ─────────────────────────────┐")
    for x, y in zip(X, Y):
        pred = net.predict(x)[0]
        print(f"  │  {x[0]:.0f} XOR {x[1]:.0f}  →  {pred:.6f}  (target: {y[0]:.1f})  "
              f"{'✓' if round(pred)==int(y[0]) else '✗'}   │")
    print("  └─────────────────────────────────────────────────┘\n")
