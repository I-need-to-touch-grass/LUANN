# matrix.py
# I created this file only to be used by network.py and also a further reference of matrix.lua

import math
import random
import time

random.seed(time.time())


class Matrix:
    def __init__(self, rows, cols, init_val=0.0):
        self.rows = rows
        self.cols = cols
        val = 0.0 if init_val is None else init_val
        self.data = [[val for _ in range(cols)] for _ in range(rows)]

    # Create a new matrix initialized to zero or a default value
    @classmethod
    def new(cls, rows, cols, init_val=0.0):
        return cls(rows, cols, init_val)

    # Create a matrix from a 2D Python list
    @classmethod
    def from_table(cls, tbl):
        rows = len(tbl)
        cols = len(tbl[0]) if rows > 0 else 0
        m = cls.new(rows, cols)
        for r in range(rows):
            for c in range(cols):
                m.data[r][c] = tbl[r][c]
        return m

    # He/Xavier-style uniform random initialization
    @classmethod
    def random(cls, rows, cols, scale=None):
        m = cls.new(rows, cols)
        s = scale if scale is not None else math.sqrt(2.0 / rows)
        for r in range(rows):
            for c in range(cols):
                m.data[r][c] = (random.random() * 2.0 - 1.0) * s
        return m

    # Deep copy a matrix
    def copy(self):
        m = Matrix.new(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                m.data[r][c] = self.data[r][c]
        return m

    # Matrix multiplication: self (R x K) * other (K x C) -> (R x C)
    def dot(self, other):
        assert self.cols == other.rows, (
            f"Dimension mismatch in dot product: "
            f"({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})"
        )
        res = Matrix.new(self.rows, other.cols)
        for r in range(self.rows):
            row_a = self.data[r]
            row_res = res.data[r]
            for k in range(self.cols):
                val_a = row_a[k]
                if val_a != 0:
                    row_b = other.data[k]
                    for c in range(other.cols):
                        row_res[c] = row_res[c] + val_a * row_b[c]
        return res

    # Transpose matrix
    def transpose(self):
        res = Matrix.new(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[c][r] = self.data[r][c]
        return res

    # Element-wise addition (supports broadcasting a 1xC row vector)
    def add(self, other):
        res = Matrix.new(self.rows, self.cols)
        if other.rows == 1 and other.cols == self.cols:
            for r in range(self.rows):
                for c in range(self.cols):
                    res.data[r][c] = self.data[r][c] + other.data[0][c]
        else:
            assert (
                self.rows == other.rows and self.cols == other.cols
            ), "Matrix add dimension mismatch"
            for r in range(self.rows):
                for c in range(self.cols):
                    res.data[r][c] = self.data[r][c] + other.data[r][c]
        return res

    # Element-wise subtraction
    def sub(self, other):
        assert (
            self.rows == other.rows and self.cols == other.cols
        ), "Matrix sub dimension mismatch"
        res = Matrix.new(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = self.data[r][c] - other.data[r][c]
        return res

    # Element-wise multiplication (Hadamard product)
    def mul(self, other):
        assert (
            self.rows == other.rows and self.cols == other.cols
        ), "Matrix mul dimension mismatch"
        res = Matrix.new(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = self.data[r][c] * other.data[r][c]
        return res

    # Scale all elements by a scalar
    def scale(self, scalar):
        res = Matrix.new(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = self.data[r][c] * scalar
        return res

    # Apply a function to every element
    def map(self, fn):
        res = Matrix.new(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[r][c] = fn(self.data[r][c], r, c)
        return res

    # Sum along axis=1 (collapse rows into a 1xC row vector)
    def sum_rows(self):
        res = Matrix.new(1, self.cols, 0.0)
        for r in range(self.rows):
            for c in range(self.cols):
                res.data[0][c] = res.data[0][c] + self.data[r][c]
        return res

    # ReLU Activation
    def relu(self):
        return self.map(lambda x, r, c: x if x > 0 else 0.0)

    # Derivative of ReLU
    def relu_deriv(self):
        return self.map(lambda x, r, c: 1.0 if x > 0 else 0.0)

    # Row-wise numerically stable Softmax
    def softmax(self):
        res = Matrix.new(self.rows, self.cols)
        for r in range(self.rows):
            max_val = -math.inf
            for c in range(self.cols):
                if self.data[r][c] > max_val:
                    max_val = self.data[r][c]

            sum_exp = 0.0
            for c in range(self.cols):
                e = math.exp(self.data[r][c] - max_val)
                res.data[r][c] = e
                sum_exp = sum_exp + e

            for c in range(self.cols):
                res.data[r][c] = res.data[r][c] / sum_exp
        return res

    # Return index (0-based) of max value per row
    def argmax_rows(self):
        indices = []
        for r in range(self.rows):
            best_c = 0
            best_v = self.data[r][0]
            for c in range(1, self.cols):
                if self.data[r][c] > best_v:
                    best_v = self.data[r][c]
                    best_c = c
            indices.append(best_c)
        return indices

