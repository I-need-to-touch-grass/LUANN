# network.py
# Feed-Forward Neural Network (v2: Adam Optimizer, Dropout, Leaky ReLU, Grad Clipping)

import math
from matrix import Matrix


def _cfg(config, key, default):
    if config is not None and key in config:
        return config[key]
    return default


class Network:
    def __init__(self, layer_sizes, config=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Hyperparameters
        self.lr = _cfg(config, "lr", 0.005)
        self.beta1 = _cfg(config, "beta1", 0.9)
        self.beta2 = _cfg(config, "beta2", 0.999)
        self.eps = 1e-8
        self.l2_reg = _cfg(config, "l2_reg", 0.0005)
        self.dropout = _cfg(config, "dropout", 0.15)
        self.clip_norm = _cfg(config, "clip_norm", 5.0)
        self.leaky_alpha = _cfg(config, "leaky_alpha", 0.01)

        self.t = 0  # Adam timestep counter
        self.weights = []
        self.biases = []

        # Adam first moment (m) and second moment (v) buffers
        self.m_w, self.v_w = [], []
        self.m_b, self.v_b = [], []

        for i in range(self.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            self.weights.append(Matrix.random(in_dim, out_dim))
            self.biases.append(Matrix.new(1, out_dim, 0.01))

            self.m_w.append(Matrix.new(in_dim, out_dim, 0.0))
            self.v_w.append(Matrix.new(in_dim, out_dim, 0.0))
            self.m_b.append(Matrix.new(1, out_dim, 0.0))
            self.v_b.append(Matrix.new(1, out_dim, 0.0))

    @classmethod
    def new(cls, layer_sizes, config=None):
        return cls(layer_sizes, config)

    # Forward pass with optional Inverted Dropout during training
    def forward(self, X, is_training=False):
        self.cache_a = [X]
        self.cache_z = []
        self.drop_masks = {}

        current = X
        for i in range(self.num_layers):
            z = current.dot(self.weights[i]).add(self.biases[i])
            self.cache_z.append(z)
            if i < self.num_layers - 1:
                current = z.leaky_relu(self.leaky_alpha)
                if is_training and self.dropout > 0:
                    mask = Matrix.dropout_mask(current.rows, current.cols, self.dropout)
                    self.drop_masks[i] = mask
                    current = current.mul(mask)
            else:
                current = z.softmax()
            self.cache_a.append(current)

        return current

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

        reg_loss = 0.0
        if self.l2_reg > 0:
            for i in range(self.num_layers):
                w = self.weights[i]
                for r in range(w.rows):
                    for c in range(w.cols):
                        reg_loss = reg_loss + (w.data[r][c] ** 2)
            reg_loss = 0.5 * self.l2_reg * reg_loss

        return data_loss + reg_loss

    # Backward pass with Adam Optimizer, Gradient Clipping & Dropout Backprop
    def backward(self, Y_true):
        batch_size = Y_true.rows
        inv_batch = 1.0 / batch_size
        self.t = self.t + 1

        bias_corr1 = 1.0 - (self.beta1 ** self.t)
        bias_corr2 = 1.0 - (self.beta2 ** self.t)

        delta = self.cache_a[self.num_layers].sub(Y_true).scale(inv_batch)

        for i in range(self.num_layers - 1, -1, -1):
            a_prev = self.cache_a[i]
            grad_w = a_prev.transpose().dot(delta)
            grad_b = delta.sum_rows()

            if self.l2_reg > 0:
                grad_w = grad_w.add(self.weights[i].scale(self.l2_reg))

            # Clip gradients to prevent spikes
            grad_w = grad_w.clip(-self.clip_norm, self.clip_norm)
            grad_b = grad_b.clip(-self.clip_norm, self.clip_norm)

            # Propagate error to previous hidden layer
            if i > 0:
                d_hidden = delta.dot(self.weights[i].transpose())
                if (i - 1) in self.drop_masks:
                    d_hidden = d_hidden.mul(self.drop_masks[i - 1])
                d_act = self.cache_z[i - 1].leaky_relu_deriv(self.leaky_alpha)
                delta = d_hidden.mul(d_act)

            # Adam moment updates for weights
            self.m_w[i] = self.m_w[i].scale(self.beta1).add(grad_w.scale(1.0 - self.beta1))
            self.v_w[i] = self.v_w[i].scale(self.beta2).add(grad_w.mul(grad_w).scale(1.0 - self.beta2))

            m_w_hat = self.m_w[i].scale(1.0 / bias_corr1)
            v_w_hat = self.v_w[i].scale(1.0 / bias_corr2)
            step_w = m_w_hat.div(v_w_hat.sqrt().add_scalar(self.eps)).scale(self.lr)
            self.weights[i] = self.weights[i].sub(step_w)

            # Adam moment updates for biases
            self.m_b[i] = self.m_b[i].scale(self.beta1).add(grad_b.scale(1.0 - self.beta1))
            self.v_b[i] = self.v_b[i].scale(self.beta2).add(grad_b.mul(grad_b).scale(1.0 - self.beta2))

            m_b_hat = self.m_b[i].scale(1.0 / bias_corr1)
            v_b_hat = self.v_b[i].scale(1.0 / bias_corr2)
            step_b = m_b_hat.div(v_b_hat.sqrt().add_scalar(self.eps)).scale(self.lr)
            self.biases[i] = self.biases[i].sub(step_b)

    def predict(self, X):
        probs = self.forward(X, is_training=False)
        indices = probs.argmax_rows()
        preds = []
        confidences = []

        for r in range(len(indices)):
            idx = indices[r]
            preds.append(idx)
            confidences.append(probs.data[r][idx])

        return preds, confidences, probs

    def evaluate(self, X, Y_true, labels):
        probs = self.forward(X, is_training=False)
        loss = self.compute_loss(probs, Y_true)
        indices = probs.argmax_rows()
        correct = 0

        for i in range(len(labels)):
            if indices[i] == labels[i]:
                correct = correct + 1

        return correct / len(labels), loss

