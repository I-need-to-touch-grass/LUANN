# Pure Lua Handwritten Digit Classifier (8x8 MLP)

A zero-dependency, pure Lua implementation of a Multi-Layer Perceptron (MLP) neural network trained to classify 8x8 grayscale handwritten digits (`0` through `9`). Everything—from matrix multiplication and backpropagation to synthetic dataset augmentation and ASCII terminal visualization—is built from scratch using only Lua's standard library.

## Directory Structure

```text
lua_digit_classifier/
├── README.md            # Project documentation (this file)
├── main.lua             # CLI runner, training loop & interactive testing
├── matrix.lua           # 2D linear algebra & activation functions
├── network.lua          # Feed-forward neural network, backprop & optimizer
├── dataset.lua          # 8x8 digit prototypes, augmentation & batching
├── serializer.lua       # Model persistence (save/load weights to disk)
├── visualizer.lua       # Terminal ASCII digit renderer & confusion matrix
└── references/          # Python reference implementations
    ├── README.md        # Detailed guide to the Python reference modules
    ├── matrix.py        # Pure Python reference for matrix.lua
    └── network.py       # Pure Python reference for network.lua
```

## Reference Implementation Architecture

The core mathematical and neural network engines (`matrix.lua` and `network.lua`) are designed alongside pure Python reference implementations located in the `references/` directory (`references/matrix.py` and `references/network.py`).

* **1-to-1 Structural Parity:** Both the Lua modules and the Python references share identical class/table structures, method names (`forward`, `backward`, `compute_loss`, `dot`, `softmax`, etc.), and internal state variables (`weights`, `biases`, `vel_w`, `vel_b`, `cache_a`, `cache_z`).
* **Cross-Language Verification:** Because Lua uses 1-based indexing and Python uses 0-based indexing, `references/matrix.py` and `references/network.py` serve as readable baselines for verifying matrix dimensions, broadcasting rules, gradient propagation, and SGD momentum updates.

---

## Module Breakdown & Core Concepts

### `matrix.lua`
Provides the foundational 2D matrix data structure and linear algebra operations needed for neural network computation without external C bindings or libraries.
* **Core Concepts:** Matrix multiplication (`dot`), transposition (`transpose`), row-vector bias broadcasting (`add`), Hadamard products (`mul`), and axis reductions (`sum_rows`).
* **Activations & Initialization:** Implements He/Xavier-style uniform weight initialization, rectified linear units (`relu` and `relu_deriv`), and numerically stable row-wise `softmax` using max-subtraction to prevent exponent overflow.

### `network.lua`
Implements a configurable, multi-layer feed-forward neural network (`Network.new({64, 48, 24, 10})`) trained via mini-batch gradient descent.
* **Core Concepts:** Forward propagation with activation caching (`cache_a`, `cache_z`), categorical cross-entropy loss paired with L2 weight decay regularization, and reverse-mode backpropagation using the simplified Softmax + Cross-Entropy output gradient ($\frac{P - Y}{N}$).
* **Optimization:** Uses Stochastic Gradient Descent (SGD) with classical momentum (`vel_w`, `vel_b`) to accelerate convergence across hidden layers.

### `dataset.lua`
Generates labeled training and evaluation datasets on the fly from hardcoded 8x8 pixel prototypes for digits `0`–`9`, removing the need to download or parse external binary files.
* **Core Concepts:** Synthetic data augmentation (random $\pm 1$ pixel spatial shifts, stroke intensity scaling, pixel dropout/bleed, and uniform sensor noise), one-hot label encoding, Fisher-Yates dataset shuffling, and mini-batch matrix slicing.

### `serializer.lua`
Handles model persistence by serializing network architecture and floating-point parameters into a human-readable plain-text format (`LUA_MLP_V1`).
* **Core Concepts:** File I/O state serialization, header validation, high-precision float formatting (`%.8f`), and reconstructing a runnable `Network` instance from disk.

### `visualizer.lua`
Provides terminal-based visual diagnostics so model behavior and errors can be inspected directly in the console.
* **Core Concepts:** Grayscale-to-ASCII shading (`" .:-=+*#@"`), horizontal probability confidence bars for all 10 digit classes, and a $10 \times 10$ confusion matrix comparing true labels against predicted classes.

### `main.lua`
Acts as the command-line entry point that orchestrates the end-to-end machine learning pipeline.
* **Core Concepts:** Hyperparameter configuration (`CONFIG`), mini-batch training loops with exponential learning rate decay (`lr_decay`), serialization round-trip verification, and an interactive REPL prompt for testing custom or random noisy digits.

---

## Quick Start

1. **Install Lua** (Lua 5.1–5.5 or LuaJIT):
   ```bash
   # macOS (Homebrew)
   brew install lua

   # Ubuntu / Debian
   sudo apt install lua5.4
   ```

2. **Run the training and interactive CLI:**
   ```bash
   lua main.lua
   ```

