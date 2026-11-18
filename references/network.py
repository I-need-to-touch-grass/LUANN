# network.py
# I created this file only to use as a reference for network.lua

import math
from matrix import Matrix


class Network:
    def __init__(self, layer_sizes, config=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.lr = (config and config.get("lr")) or 0.05
        self.momentum = (config and config.get("momentum")) or 0.9
        self.l2_reg = (config and config.get("l2_reg")) or 0.0005

        self.weights = []
        self.biases = []
        self.vel_w = []
        self.vel_b = []

        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            self.weights.append(Matrix.random(in_dim, out_dim))
            self.biases.append(Matrix.new(1, out_dim, 0.01))
            self.vel_w.append(Matrix.new(in_dim, out_dim, 0.0))
            self.vel_b.append(Matrix.new(1, out_dim, 0.0))

    @classmethod
    def new(cls, layer_sizes, config=None):
        return cls(layer_sizes, config)

    # Forward pass through all layers; caches activations for backprop
    def forward(self, X):
        self.cache_a = [X]
        self.cache_z = []

        current = X
        for i in range(self.num_layers):
            z = current.dot(self.weights[i]).add(self.biases[i])
            self.cache_z.append(z)
            if i < self.num_layers - 1:
                current = z.relu()
            else:
                current = z.softmax()
            self.cache_a.append(current)

        return current

    # Compute Cross-Entropy Loss + L2 Regularization Penalty
    def compute_loss(self, probs, Y_true):
        batch_size = probs.rows
        eps = 1e-12
        data_loss = 0.0

        for r in range(batch_size):
            for c in range(probs.cols):
                if Y_true.data[r][c] > 0:
                    p = max(probs.data[r][c], eps)
                    data_loss = data_loss - Y_true.data[r][c] * math.log(p)
        data_loss = data_loss / batch_size

        # L2 regularization loss
        reg_loss = 0.0
        if self.l2_reg > 0:
            for i in range(self.num_layers):
                w = self.weights[i]
                for r in range(w.rows):
                    for c in range(w.cols):
                        reg_loss = reg_loss + (w.data[r][c] ** 2)
            reg_loss = 0.5 * self.l2_reg * reg_loss

        return data_loss + reg_loss

    # Backward pass & SGD with Momentum parameter update
    def backward(self, Y_true):
        batch_size = Y_true.rows
        inv_batch = 1.0 / batch_size

        # Gradient of Cross-Entropy + Softmax simplifies to (P - Y) / N
        delta = self.cache_a[self.num_layers].sub(Y_true).scale(inv_batch)

        for i in range(self.num_layers - 1, -1, -1):
            a_prev = self.cache_a[i]
            grad_w = a_prev.transpose().dot(delta)
            grad_b = delta.sum_rows()

            # Add L2 weight decay gradient
            if self.l2_reg > 0:
                grad_w = grad_w.add(self.weights[i].scale(self.l2_reg))

            # Propagate error to previous hidden layer before updating weights
            if i > 0:
                d_hidden = delta.dot(self.weights[i].transpose())
                d_relu = self.cache_z[i - 1].relu_deriv()
                delta = d_hidden.mul(d_relu)

            # Update velocity & apply weights
            self.vel_w[i] = self.vel_w[i].scale(self.momentum).sub(grad_w.scale(self.lr))
            self.vel_b[i] = self.vel_b[i].scale(self.momentum).sub(grad_b.scale(self.lr))

            self.weights[i] = self.weights[i].add(self.vel_w[i])
            self.biases[i] = self.biases[i].add(self.vel_b[i])

    # Predict digit class (0-9) and confidence scores for input matrix X
    def predict(self, X):
        probs = self.forward(X)
        indices = probs.argmax_rows()
        preds = []
        confidences = []

        for r in range(len(indices)):
            idx = indices[r]
            preds.append(idx)  # 0-based index already matches 0-9 digit
            confidences.append(probs.data[r][idx])

        return preds, confidences, probs

    # Evaluate accuracy and average loss on a full dataset matrix pair
    def evaluate(self, X, Y_true, labels):
        probs = self.forward(X)
        loss = self.compute_loss(probs, Y_true)
        indices = probs.argmax_rows()
        correct = 0

        for i in range(len(labels)):
            if indices[i] == labels[i]:
                correct = correct + 1

        return correct / len(labels), loss

