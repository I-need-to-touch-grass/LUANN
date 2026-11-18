# Python Reference Implementations (`references/`)

This directory contains zero-dependency, pure Python reference implementations of the core math and neural network modules used in the main Lua project:

* `matrix.py` — Reference implementation for `../matrix.lua`
* `network.py` — Reference implementation for `../network.lua`

These files mirror the exact class names, method signatures, and internal variable names of their Lua counterparts (`Matrix`, `Network`, `weights`, `biases`, `vel_w`, `vel_b`, `cache_a`, `cache_z`) to make cross-language debugging, algorithm verification, and prototyping straightforward.

---

## `matrix.py` — Pure Python 2D Matrix & Math Engine

`matrix.py` implements a self-contained `Matrix` class backed by standard Python nested lists (`list[list[float]]`) and Python's built-in `math` and `random` modules (no NumPy required).

### Key Data Attributes
* `rows` (`int`): Number of rows in the matrix.
* `cols` (`int`): Number of columns in the matrix.
* `data` (`list[list[float]]`): 2D array storing matrix elements in row-major order.

### Constructors & Initializers
* `Matrix(rows, cols, init_val=0.0)` / `Matrix.new(...)`: Allocates a `rows x cols` matrix filled with `init_val`.
* `Matrix.from_table(tbl)`: Converts a 2D Python list into a `Matrix` instance.
* `Matrix.random(rows, cols, scale=None)`: Initializes weights from a uniform distribution $[-\text{scale}, +\text{scale}]$, defaulting to He-style scaling ($\sqrt{\frac{2}{\text{rows}}}$).
* `copy()`: Returns a deep copy of the matrix.

### Linear Algebra & Element-Wise Operations
* `dot(other)`: Performs standard matrix multiplication $(R \times K) \cdot (K \times C) \rightarrow (R \times C)$ with a zero-skip optimization (`if val_a != 0`) for sparse inputs.
* `transpose()`: Swaps rows and columns $(R \times C) \rightarrow (C \times R)$.
* `add(other)`: Element-wise addition. Automatically broadcasts a `1 x C` row vector across all `R` rows when adding bias vectors.
* `sub(other)` / `mul(other)`: Element-wise subtraction and Hadamard (element-wise) multiplication.
* `scale(scalar)`: Multiplies every element by a scalar float.
* `map(fn)`: Applies a callback `fn(val, r, c)` across every element and returns a new `Matrix`.
* `sum_rows()`: Collapses a `R x C` matrix along the row axis into a `1 x C` row vector (used to compute bias gradients across a mini-batch).

### Activations & Predictions
* `relu()` / `relu_deriv()`: Applies the Rectified Linear Unit activation $\max(0, x)$ and its piecewise derivative ($1$ if $x > 0$, else $0$).
* `softmax()`: Computes row-wise probabilities via $\frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$, subtracting the row maximum first for numerical stability.
* `argmax_rows()`: Returns a list of 0-based column indices corresponding to the highest probability in each row.

---

## `network.py` — Multi-Layer Perceptron & Backpropagation Engine

`network.py` imports `Matrix` from `matrix.py` and implements a fully connected feed-forward neural network trained with backpropagation and SGD with momentum.

### Key Attributes
* `layer_sizes` (`list[int]`): Topology specification (e.g., `[64, 48, 24, 10]`).
* `num_layers` (`int`): Number of weighted transformations (`len(layer_sizes) - 1`).
* `lr` (`float`): Learning rate (default `0.05`).
* `momentum` (`float`): Momentum decay coefficient (default `0.9`).
* `l2_reg` (`float`): L2 weight decay penalty coefficient (default `0.0005`).
* `weights` / `biases`: Lists of `Matrix` objects storing trainable parameters for each layer.
* `vel_w` / `vel_b`: Velocity buffers storing exponentially weighted moving averages of parameter updates.
* `cache_a` / `cache_z`: Intermediate post-activation (`A`) and pre-activation (`Z`) matrices saved during `forward()` for reuse in `backward()`.

### Core Methods
* `forward(X)`: Passes an input batch `X` (`batch_size x input_dim`) through all layers. Uses `relu()` on hidden layers and `softmax()` on the final output layer.
* `compute_loss(probs, Y_true)`: Computes total loss as the sum of average categorical cross-entropy loss ($-\frac{1}{N}\sum Y \log(\max(P, 10^{-12}))$) and L2 regularization loss ($\frac{1}{2}\lambda \sum W^2$).
* `backward(Y_true)`: Computes gradients from output to input:
  1. Starts with the combined Softmax + Cross-Entropy error signal $\delta^{(L)} = \frac{P - Y}{N}$.
  2. Computes weight gradients (`a_prev.transpose().dot(delta)`) plus L2 decay (`l2_reg * W`), and bias gradients (`delta.sum_rows()`).
  3. Propagates $\delta$ backward through hidden layers via $\delta^{(l)} = (\delta^{(l+1)} W^{(l+1)T}) \odot \text{ReLU}'(Z^{(l)})$.
  4. Updates velocity buffers ($v = \mu v - \eta \nabla$) and applies parameter steps.
* `predict(X)`: Runs a forward pass and returns `(preds, confidences, probs)` where `preds` are 0-based digit labels (`0`–`9`).
* `evaluate(X, Y_true, labels)`: Returns `(accuracy, loss)` across an evaluation dataset.

---

## Lua vs. Python Differences

When comparing `references/*.py` against `../*.lua`, keep these language conventions in mind:

| Feature | Lua (`../*.lua`) | Python (`references/*.py`) |
| :--- | :--- | :--- |
| **Indexing** | 1-based (`1` to `N`) | 0-based (`0` to `N - 1`) |
| **Predicted Class Mapping** | `argmax_rows()` returns `1..10`; subtracted by `1` in `predict()` | `argmax_rows()` directly returns `0..9` |
| **Activation Cache** | `cache_a` stores input `X`; `cache_a[1..L]` stores layer outputs | `cache_a` stores input `X`; `cache_a[1..L]` stores layer outputs |
| **Constructors** | `Matrix.new(...)` and `Network.new(...)` | Standard `__init__` plus `.new(...)` classmethods for API parity |

## Running a Standalone Sanity Check

From inside the `references/` directory, you can verify both modules together using Python 3:

```bash
cd references
python3 -c "from matrix import Matrix; from network import Network; net = Network([64, 32, 10]); print(net.predict(Matrix.random(2, 64))[:2])"
```

