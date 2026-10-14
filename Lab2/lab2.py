import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

X, y_raw = load_digits(return_X_y=True)
X = X / 16.0
y = np.zeros((y_raw.size, 10))
y[np.arange(y_raw.size), y_raw] = 1.0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

def relu(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

class NeuralNetwork:
    def __init__(self, W):
        np.random.seed(42)
        self.W1 = np.random.randn(64, W) * 0.1
        self.b1 = np.zeros((1, W))
        self.W2 = np.random.randn(W, W) * 0.1
        self.b2 = np.zeros((1, W))
        self.W3 = np.random.randn(W, 10) * 0.1
        self.b3 = np.zeros((1, 10))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = relu(self.Z2)
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = softmax(self.Z3)
        return self.A3

    def compute_loss(self, Y_pred, Y_true):
        return -np.sum(Y_true * np.log(Y_pred + 1e-9)) / Y_true.shape[0]

    def backward(self, X, Y_true):
        m = X.shape[0]
        dZ3 = self.A3 - Y_true
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * relu_deriv(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2, dW3, db3

    def update(self, grads, lr):
        dW1, db1, dW2, db2, dW3, db3 = grads
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

def gradient_check():
    nn = NeuralNetwork(W=16)
    X_sample, Y_sample = X_train[:5], y_train[:5]
    nn.forward(X_sample)
    grads = nn.backward(X_sample, Y_sample)
    dW1_analytical = grads[0]
    
    epsilon = 1e-5
    passed = 0
    np.random.seed(42)
    indices = [(np.random.randint(64), np.random.randint(16)) for _ in range(10)]
    
    for i, j in indices:
        orig = nn.W1[i, j]
        
        nn.W1[i, j] = orig + epsilon
        loss_plus = nn.compute_loss(nn.forward(X_sample), Y_sample)
        
        nn.W1[i, j] = orig - epsilon
        loss_minus = nn.compute_loss(nn.forward(X_sample), Y_sample)
        
        nn.W1[i, j] = orig
        dW1_num = (loss_plus - loss_minus) / (2 * epsilon)
        
        rel_error = abs(dW1_analytical[i, j] - dW1_num) / (abs(dW1_analytical[i, j]) + abs(dW1_num) + 1e-9)
        if rel_error < 1e-6:
            passed += 1
            
    print(f"Gradient Check: {passed}/10 passed")

gradient_check()

widths = [16, 64]
learning_rates = [0.001, 0.01, 0.1]
epochs = 50
batch_size = 32

os.makedirs('results', exist_ok=True)
plt.figure(figsize=(10, 6))

for W in widths:
    for lr in learning_rates:
        nn = NeuralNetwork(W=W)
        m = X_train.shape[0]
        loss_history = []
        
        for epoch in range(epochs):
            indices = np.arange(m)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                nn.forward(X_batch)
                grads = nn.backward(X_batch, y_batch)
                nn.update(grads, lr)
            
            epoch_loss = nn.compute_loss(nn.forward(X_train), y_train)
            loss_history.append(epoch_loss)
            
        test_preds = nn.forward(X_test)
        acc = np.mean(np.argmax(test_preds, axis=1) == np.argmax(y_test, axis=1)) * 100
        print(f"W={W}, LR={lr} | Final Test Accuracy: {acc:.2f}%")
        
        plt.plot(loss_history, label=f"W={W}, LR={lr}")

plt.title("Training Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig('results/loss_curves.png')

nn_timing = NeuralNetwork(W=64)
X_time = X_train[:32]
y_time = y_train[:32]

fwd_times = []
for _ in range(100):
    start = time.perf_counter()
    nn_timing.forward(X_time)
    fwd_times.append(time.perf_counter() - start)

bwd_times = []
nn_timing.forward(X_time)
for _ in range(100):
    start = time.perf_counter()
    nn_timing.backward(X_time, y_time)
    bwd_times.append(time.perf_counter() - start)

print(f"Median Forward Time: {np.median(fwd_times):.6f} seconds")
print(f"Median Backward Time: {np.median(bwd_times):.6f} seconds")
plt.show()